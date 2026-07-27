from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from codecraft.retrieval.chunking import CodeChunk, CodeSymbol, TreeSitterChunker
from codecraft.retrieval.errors import RetrievalUnavailableError
from codecraft.retrieval.files import (
    is_inside_workspace,
    iter_workspace_files,
    looks_binary,
)

INDEX_SCHEMA_VERSION = 1
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9_]+|[^\W_]+")
_STOP_WORDS = frozenset({"a", "an", "are", "is", "of", "the", "to", "where"})


@dataclass(frozen=True, slots=True)
class IndexSyncStats:
    candidate_file_count: int
    indexed_file_count: int
    updated_file_count: int
    unchanged_file_count: int
    deleted_file_count: int
    chunk_count: int
    symbol_count: int
    skipped_binary_count: int
    skipped_large_count: int
    indexed_bytes: int
    database_path: str


@dataclass(frozen=True, slots=True)
class IndexedMatch:
    path: str
    line: int
    snippet: str


@dataclass(frozen=True, slots=True)
class IndexQueryResult:
    matches: tuple[IndexedMatch, ...]
    indexed_file_count: int
    stale_file_count: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class _FileRefreshResult:
    updated_file_count: int = 0
    unchanged_file_count: int = 0
    deleted_file_count: int = 0
    skipped_binary_count: int = 0
    skipped_large_count: int = 0
    indexed_bytes: int = 0


class _IndexRow(TypedDict):
    path: str
    line: int
    snippet: str
    mtime_ns: int
    size: int


class RepositoryIndex:
    def __init__(self, index_root: Path, *, chunker: TreeSitterChunker | None = None):
        self.index_root = index_root.expanduser().resolve()
        self._chunker = chunker

    @property
    def chunker(self) -> TreeSitterChunker:
        if self._chunker is None:
            self._chunker = TreeSitterChunker()
        return self._chunker

    def database_path(self, workspace_root: Path) -> Path:
        root = workspace_root.expanduser().resolve()
        workspace_id = hashlib.sha256(str(root).encode()).hexdigest()[:24]
        return self.index_root / workspace_id / "index.sqlite3"

    def sync(
        self,
        workspace_root: Path,
        *,
        max_file_bytes: int = 1_000_000,
    ) -> IndexSyncStats:
        root = workspace_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"workspace root must be a directory: {root}")
        database = self.database_path(root)
        database.parent.mkdir(parents=True, exist_ok=True)
        files = [
            path
            for path in iter_workspace_files(root)
            if not is_inside_workspace(path, (self.index_root,))
        ]
        updated = 0
        unchanged = 0
        deleted = 0
        skipped_binary = 0
        skipped_large = 0
        indexed_bytes = 0

        with closing(self._connect(database)) as connection:
            self._initialize(connection, root)
            known = {
                row["path"]: (row["mtime_ns"], row["size"], row["digest"])
                for row in connection.execute(
                    "SELECT path, mtime_ns, size, digest FROM files"
                )
            }
            visible_paths: set[str] = set()
            for file_path in files:
                relative = str(file_path.relative_to(root))
                visible_paths.add(relative)
                stat = file_path.stat()
                if stat.st_size > max_file_bytes:
                    skipped_large += 1
                    self._delete_file(connection, relative)
                    continue
                previous = known.get(relative)
                if previous and previous[:2] == (stat.st_mtime_ns, stat.st_size):
                    unchanged += 1
                    continue
                try:
                    raw = file_path.read_bytes()
                except OSError:
                    continue
                if looks_binary(raw):
                    skipped_binary += 1
                    self._delete_file(connection, relative)
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if previous and previous[2] == digest:
                    connection.execute(
                        "UPDATE files SET mtime_ns = ?, size = ? WHERE path = ?",
                        (stat.st_mtime_ns, stat.st_size, relative),
                    )
                    unchanged += 1
                    continue

                content = raw.decode("utf-8", errors="replace")
                chunked = self.chunker.chunk(file_path, content)
                self._replace_file(
                    connection,
                    path=relative,
                    mtime_ns=stat.st_mtime_ns,
                    size=stat.st_size,
                    digest=digest,
                    language=chunked.language,
                    chunks=chunked.chunks,
                    symbols=chunked.symbols,
                )
                updated += 1
                indexed_bytes += len(raw)

            for missing in known.keys() - visible_paths:
                self._delete_file(connection, missing)
                deleted += 1
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('indexed_at', ?)",
                (datetime.now(UTC).isoformat(),),
            )
            connection.commit()
            indexed_file_count = connection.execute(
                "SELECT COUNT(*) FROM files"
            ).fetchone()[0]
            chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[
                0
            ]
            symbol_count = connection.execute(
                "SELECT COUNT(*) FROM symbols"
            ).fetchone()[0]

        return IndexSyncStats(
            candidate_file_count=len(files),
            indexed_file_count=indexed_file_count,
            updated_file_count=updated,
            unchanged_file_count=unchanged,
            deleted_file_count=deleted,
            chunk_count=chunk_count,
            symbol_count=symbol_count,
            skipped_binary_count=skipped_binary,
            skipped_large_count=skipped_large,
            indexed_bytes=indexed_bytes,
            database_path=str(database),
        )

    def refresh_paths(
        self,
        workspace_root: Path,
        paths: list[Path],
        *,
        max_file_bytes: int = 1_000_000,
    ) -> IndexSyncStats:
        root = workspace_root.expanduser().resolve()
        database = self.database_path(root)
        if not database.is_file():
            raise RetrievalUnavailableError("repository index has not been built")

        selected = self._select_refresh_paths(root, paths)
        root, connection = self._open_existing(root)
        with closing(connection):
            results = [
                self._refresh_file(
                    connection,
                    root,
                    file_path,
                    max_file_bytes=max_file_bytes,
                )
                for file_path in selected
            ]

            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('indexed_at', ?)",
                (datetime.now(UTC).isoformat(),),
            )
            connection.commit()
            indexed_file_count = connection.execute(
                "SELECT COUNT(*) FROM files"
            ).fetchone()[0]
            chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[
                0
            ]
            symbol_count = connection.execute(
                "SELECT COUNT(*) FROM symbols"
            ).fetchone()[0]

        return IndexSyncStats(
            candidate_file_count=len(selected),
            indexed_file_count=indexed_file_count,
            updated_file_count=sum(item.updated_file_count for item in results),
            unchanged_file_count=sum(item.unchanged_file_count for item in results),
            deleted_file_count=sum(item.deleted_file_count for item in results),
            chunk_count=chunk_count,
            symbol_count=symbol_count,
            skipped_binary_count=sum(item.skipped_binary_count for item in results),
            skipped_large_count=sum(item.skipped_large_count for item in results),
            indexed_bytes=sum(item.indexed_bytes for item in results),
            database_path=str(database),
        )

    def _select_refresh_paths(self, root: Path, paths: list[Path]) -> list[Path]:
        selected: list[Path] = []
        for path in paths:
            candidate = path.expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve(strict=False)
            if not is_inside_workspace(candidate, (root,)):
                continue
            if is_inside_workspace(candidate, (self.index_root,)):
                continue
            if candidate not in selected:
                selected.append(candidate)
        return selected

    def _refresh_file(
        self,
        connection: sqlite3.Connection,
        root: Path,
        file_path: Path,
        *,
        max_file_bytes: int,
    ) -> _FileRefreshResult:
        relative = str(file_path.relative_to(root))
        previous = connection.execute(
            "SELECT mtime_ns, size, digest FROM files WHERE path = ?",
            (relative,),
        ).fetchone()
        if not file_path.is_file():
            self._delete_file(connection, relative)
            return _FileRefreshResult(
                deleted_file_count=int(previous is not None),
            )

        stat = file_path.stat()
        if stat.st_size > max_file_bytes:
            self._delete_file(connection, relative)
            return _FileRefreshResult(skipped_large_count=1)
        if previous and (previous["mtime_ns"], previous["size"]) == (
            stat.st_mtime_ns,
            stat.st_size,
        ):
            return _FileRefreshResult(unchanged_file_count=1)

        raw = file_path.read_bytes()
        if looks_binary(raw):
            self._delete_file(connection, relative)
            return _FileRefreshResult(skipped_binary_count=1)

        digest = hashlib.sha256(raw).hexdigest()
        if previous and previous["digest"] == digest:
            connection.execute(
                "UPDATE files SET mtime_ns = ?, size = ? WHERE path = ?",
                (stat.st_mtime_ns, stat.st_size, relative),
            )
            return _FileRefreshResult(unchanged_file_count=1)

        content = raw.decode("utf-8", errors="replace")
        chunked = self.chunker.chunk(file_path, content)
        self._replace_file(
            connection,
            path=relative,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            digest=digest,
            language=chunked.language,
            chunks=chunked.chunks,
            symbols=chunked.symbols,
        )
        return _FileRefreshResult(
            updated_file_count=1,
            indexed_bytes=len(raw),
        )

    def search_lexical(
        self,
        workspace_root: Path,
        *,
        query: str,
        scope: str = ".",
        mode: str = "both",
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> IndexQueryResult:
        root, connection = self._open_existing(workspace_root)
        with closing(connection):
            indexed_file_count = connection.execute(
                "SELECT COUNT(*) FROM files"
            ).fetchone()[0]
            if mode == "path":
                rows = self._path_rows(
                    connection,
                    query=query,
                    scope=scope,
                    case_sensitive=case_sensitive,
                    limit=max_results * 3 + 1,
                )
            else:
                if case_sensitive:
                    raise RetrievalUnavailableError(
                        "FTS5 lexical retrieval is case-insensitive"
                    )
                expression = _fts_expression(query)
                if not expression:
                    raise RetrievalUnavailableError("query has no indexable terms")
                rows = self._lexical_rows(
                    connection,
                    expression=expression,
                    scope=scope,
                    limit=max_results * 3 + 1,
                )
            return self._validated_result(
                root,
                rows,
                indexed_file_count=indexed_file_count,
                max_results=max_results,
            )

    def search_symbols(
        self,
        workspace_root: Path,
        *,
        query: str,
        scope: str = ".",
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> IndexQueryResult:
        root, connection = self._open_existing(workspace_root)
        comparison = "name = ?" if case_sensitive else "name = ? COLLATE NOCASE"
        prefix = "name LIKE ?" if case_sensitive else "name LIKE ? COLLATE NOCASE"
        scope_sql, scope_args = _scope_clause(scope, column="symbols.path")
        sql = f"""
            SELECT symbols.path, symbols.line, symbols.signature AS snippet,
                   files.mtime_ns, files.size
            FROM symbols JOIN files ON files.path = symbols.path
            WHERE ({comparison} OR {prefix}) {scope_sql}
            ORDER BY CASE WHEN {comparison} THEN 0 ELSE 1 END,
                     symbols.path, symbols.line
            LIMIT ?
        """
        args = [query, f"{query}%", *scope_args, query, max_results * 3 + 1]
        with closing(connection):
            rows = connection.execute(sql, args).fetchall()
            indexed_file_count = connection.execute(
                "SELECT COUNT(*) FROM files"
            ).fetchone()[0]
            return self._validated_result(
                root,
                rows,
                indexed_file_count=indexed_file_count,
                max_results=max_results,
            )

    @staticmethod
    def _connect(database: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(database, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _open_existing(self, workspace_root: Path) -> tuple[Path, sqlite3.Connection]:
        root = workspace_root.expanduser().resolve()
        database = self.database_path(root)
        if not database.is_file():
            raise RetrievalUnavailableError(
                f"repository index does not exist: {database}"
            )
        connection = self._connect(database)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            stored_root = connection.execute(
                "SELECT value FROM metadata WHERE key = 'workspace_root'"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            connection.close()
            raise RetrievalUnavailableError(
                f"repository index is invalid: {exc}"
            ) from exc
        if version != INDEX_SCHEMA_VERSION or stored_root is None:
            connection.close()
            raise RetrievalUnavailableError("repository index schema is incompatible")
        if stored_root[0] != str(root):
            connection.close()
            raise RetrievalUnavailableError(
                "repository index belongs to another workspace"
            )
        return root, connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection, root: Path) -> None:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files(
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                digest TEXT NOT NULL,
                language TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                kind TEXT NOT NULL,
                symbol TEXT,
                content TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                path, symbol, content, tokenize = 'unicode61'
            );
            CREATE TABLE IF NOT EXISTS symbols(
                path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                line INTEGER NOT NULL,
                signature TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);
            """
        )
        connection.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('workspace_root', ?)",
            (str(root),),
        )

    def _replace_file(
        self,
        connection: sqlite3.Connection,
        *,
        path: str,
        mtime_ns: int,
        size: int,
        digest: str,
        language: str,
        chunks: tuple[CodeChunk, ...],
        symbols: tuple[CodeSymbol, ...],
    ) -> None:
        self._delete_file(connection, path)
        connection.execute(
            "INSERT INTO files(path, mtime_ns, size, digest, language) VALUES(?, ?, ?, ?, ?)",
            (path, mtime_ns, size, digest, language),
        )
        for chunk in chunks:
            cursor = connection.execute(
                """INSERT INTO chunks(path, start_line, end_line, kind, symbol, content)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (
                    path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.kind,
                    chunk.symbol,
                    chunk.content,
                ),
            )
            connection.execute(
                "INSERT INTO chunk_fts(rowid, path, symbol, content) VALUES(?, ?, ?, ?)",
                (cursor.lastrowid, path, chunk.symbol or "", chunk.content),
            )
        connection.executemany(
            "INSERT INTO symbols(path, name, kind, line, signature) VALUES(?, ?, ?, ?, ?)",
            [
                (path, symbol.name, symbol.kind, symbol.line, symbol.signature)
                for symbol in symbols
            ],
        )

    @staticmethod
    def _delete_file(connection: sqlite3.Connection, path: str) -> None:
        connection.execute(
            "DELETE FROM chunk_fts WHERE rowid IN (SELECT id FROM chunks WHERE path = ?)",
            (path,),
        )
        connection.execute("DELETE FROM symbols WHERE path = ?", (path,))
        connection.execute("DELETE FROM chunks WHERE path = ?", (path,))
        connection.execute("DELETE FROM files WHERE path = ?", (path,))

    @staticmethod
    def _lexical_rows(
        connection: sqlite3.Connection,
        *,
        expression: str,
        scope: str,
        limit: int,
    ) -> list[_IndexRow]:
        scope_sql, scope_args = _scope_clause(scope, column="chunks.path")
        sql = f"""
            SELECT chunks.path, chunks.start_line AS line, chunks.content,
                   files.mtime_ns, files.size
            FROM chunk_fts
            JOIN chunks ON chunks.id = chunk_fts.rowid
            JOIN files ON files.path = chunks.path
            WHERE chunk_fts MATCH ? {scope_sql}
            ORDER BY bm25(chunk_fts), chunks.path, chunks.start_line
            LIMIT ?
        """
        rows = connection.execute(sql, [expression, *scope_args, limit]).fetchall()
        results: list[_IndexRow] = []
        for row in rows:
            line_offset, snippet = _matching_line(str(row["content"]), expression)
            results.append(
                _IndexRow(
                    path=str(row["path"]),
                    line=int(row["line"]) + line_offset,
                    snippet=snippet,
                    mtime_ns=int(row["mtime_ns"]),
                    size=int(row["size"]),
                )
            )
        return results

    @staticmethod
    def _path_rows(
        connection: sqlite3.Connection,
        *,
        query: str,
        scope: str,
        case_sensitive: bool,
        limit: int,
    ) -> list[_IndexRow]:
        scope_sql, scope_args = _scope_clause(scope, column="files.path")
        predicate = (
            "instr(files.path, ?) > 0"
            if case_sensitive
            else "instr(lower(files.path), lower(?)) > 0"
        )
        sql = f"""
            SELECT files.path, 1 AS line, files.path AS snippet,
                   files.mtime_ns, files.size
            FROM files WHERE {predicate} {scope_sql}
            ORDER BY files.path LIMIT ?
        """
        rows = connection.execute(sql, [query, *scope_args, limit]).fetchall()
        return [
            _IndexRow(
                path=str(row["path"]),
                line=int(row["line"]),
                snippet=str(row["snippet"]),
                mtime_ns=int(row["mtime_ns"]),
                size=int(row["size"]),
            )
            for row in rows
        ]

    @staticmethod
    def _validated_result(
        root: Path,
        rows: list[_IndexRow],
        *,
        indexed_file_count: int,
        max_results: int,
    ) -> IndexQueryResult:
        matches: list[IndexedMatch] = []
        stale_paths: set[str] = set()
        seen: set[tuple[str, int]] = set()
        freshness: dict[str, bool] = {}
        for row in rows:
            path = str(row["path"])
            if path not in freshness:
                try:
                    stat = (root / path).stat()
                    freshness[path] = (
                        stat.st_mtime_ns == row["mtime_ns"]
                        and stat.st_size == row["size"]
                    )
                except OSError:
                    freshness[path] = False
            if not freshness[path]:
                stale_paths.add(path)
                continue
            key = (path, int(row["line"]))
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                IndexedMatch(
                    path=path,
                    line=int(row["line"]),
                    snippet=str(row["snippet"]).strip()[:240],
                )
            )
            if len(matches) >= max_results:
                break
        return IndexQueryResult(
            matches=tuple(matches),
            indexed_file_count=indexed_file_count,
            stale_file_count=len(stale_paths),
            truncated=len(rows) > max_results,
        )


def _fts_expression(query: str) -> str:
    tokens = [
        token
        for token in _QUERY_TOKEN.findall(query)
        if token.casefold() not in _STOP_WORDS and len(token) > 1
    ]
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _scope_clause(scope: str, *, column: str) -> tuple[str, list[str]]:
    normalized = scope.strip("./")
    if not normalized:
        return "", []
    return f"AND ({column} = ? OR {column} LIKE ?)", [normalized, f"{normalized}/%"]


def _matching_line(content: str, expression: str) -> tuple[int, str]:
    terms = [term.strip('"').casefold() for term in expression.split(" OR ")]
    lines = content.splitlines()
    for offset, line in enumerate(lines):
        folded = line.casefold()
        if any(term in folded for term in terms):
            return offset, line.strip()
    for offset, line in enumerate(lines):
        if line.strip():
            return offset, line.strip()
    return 0, ""
