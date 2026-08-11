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
    """全量或增量同步后的文件、块、符号、跳过项与字节统计。"""

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
    """通过索引验证为新鲜的路径、行号和短摘要命中。"""

    path: str
    line: int
    snippet: str


@dataclass(frozen=True, slots=True)
class IndexQueryResult:
    """索引查询命中及索引规模、陈旧文件和截断事实。"""

    matches: tuple[IndexedMatch, ...]
    indexed_file_count: int
    stale_file_count: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class _FileRefreshResult:
    """单文件增量刷新对同步统计的贡献。"""

    updated_file_count: int = 0
    unchanged_file_count: int = 0
    deleted_file_count: int = 0
    skipped_binary_count: int = 0
    skipped_large_count: int = 0
    indexed_bytes: int = 0


class _IndexRow(TypedDict):
    """查询层交给新鲜度验证层的 SQLite 行形态。"""

    path: str
    line: int
    snippet: str
    mtime_ns: int
    size: int


class RepositoryIndex:
    """按 workspace 隔离的 SQLite FTS5、代码块与符号持久索引。"""

    def __init__(self, index_root: Path, *, chunker: TreeSitterChunker | None = None):
        """设置索引存储根，并可注入 Chunker 以便测试或替换切分策略。"""
        self.index_root = index_root.expanduser().resolve()
        self._chunker = chunker

    @property
    def chunker(self) -> TreeSitterChunker:
        """首次索引源码时延迟构造 TreeSitterChunker。"""
        if self._chunker is None:
            self._chunker = TreeSitterChunker()
        return self._chunker

    def database_path(self, workspace_root: Path) -> Path:
        """用 workspace 绝对路径哈希生成稳定且互相隔离的数据库路径。

        Example:
            ``RepositoryIndex(cache).database_path(repo)`` 形如
            ``cache/<24位sha256>/index.sqlite3``。
        """
        root = workspace_root.expanduser().resolve()
        workspace_id = hashlib.sha256(str(root).encode()).hexdigest()[:24]
        return self.index_root / workspace_id / "index.sqlite3"

    def sync(
        self,
        workspace_root: Path,
        *,
        max_file_bytes: int = 1_000_000,
    ) -> IndexSyncStats:
        """全量对账 workspace 文件与索引，复用未变项并删除消失项。

        快路径先比较 ``mtime_ns + size``；时间变化后再比较 SHA-256，内容
        未变时只更新元数据。文本变化才重新分块并在同一事务替换文件、FTS
        和符号记录。二进制或超大文件会从旧索引删除，避免返回过期内容。

        Args:
            workspace_root: 要建立或更新索引的仓库根。
            max_file_bytes: 单文件可索引字节上限。

        Returns:
            本次候选、更新、复用、删除、跳过和当前数据库规模。

        Raises:
            ValueError: workspace_root 不是目录。
        """
        root = workspace_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"workspace root must be a directory: {root}")
        database = self.database_path(root)
        database.parent.mkdir(parents=True, exist_ok=True)
        files = [
            path
            for path in iter_workspace_files(root)
            if not is_inside_workspace(path, self.index_root)
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
        """只对账工具已改动的安全路径，要求该 workspace 已完成首次 sync。

        Args:
            workspace_root: 初次建索引时的同一 workspace 根。
            paths: 绝对路径或相对 workspace 的潜在变更路径。
            max_file_bytes: 单文件可索引字节上限。

        Raises:
            RetrievalUnavailableError: 数据库尚未创建或不可兼容。
        """
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
        """解析、去重并过滤 workspace 外或索引存储目录内的刷新目标。"""
        selected: list[Path] = []
        for path in paths:
            candidate = path.expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve(strict=False)
            if not is_inside_workspace(candidate, root):
                continue
            if is_inside_workspace(candidate, self.index_root):
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
        """对账一个路径的存在性、大小、二进制、时间、摘要和索引正文。"""
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
        """执行 FTS5 内容检索或文件路径子串检索并过滤陈旧命中。

        Raises:
            RetrievalUnavailableError: 索引不存在/不兼容、大小写敏感内容查询
            无法由 FTS5 提供，或查询没有可索引词。
        """
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
        """按符号精确名称优先、前缀次之检索，并过滤陈旧命中。"""
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
        """打开带五秒锁等待和名称列访问的 SQLite 连接。"""
        connection = sqlite3.connect(database, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _open_existing(self, workspace_root: Path) -> tuple[Path, sqlite3.Connection]:
        """打开并验证 schema version 与 workspace 所有权元数据。

        任一验证失败都会先关闭连接，再转成 RetrievalUnavailableError，供
        ContextEngine 降级到无索引扫描。
        """
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
        """幂等创建 WAL、外键、metadata/files/chunks/FTS/symbols schema。"""
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
        """在事务内删除旧版本并原子写入文件、块、FTS 行与符号。"""
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
        """按依赖顺序删除一个文件的 FTS、符号、块和主记录。"""
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
        """执行带 scope 与 BM25 排序的 FTS 查询，并定位首个匹配行。"""
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
        """执行可选大小写敏感的文件路径子串查询。"""
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
        """按当前 stat 丢弃陈旧行、按路径行号去重并限制输出数量。

        索引查询会预取最多三倍结果，使少量陈旧/重复行被过滤后仍有机会填满
        max_results；但只要命中过陈旧文件，就计入 stale_file_count 供上层
        决定整次降级，避免混合新旧事实。
        """
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
    """提取非停用词并构造安全的 FTS5 quoted OR 表达式。

    Example:
        >>> _fts_expression("where is payment_timeout_ms")
        '"payment_timeout_ms"'
    """
    tokens = [
        token
        for token in _QUERY_TOKEN.findall(query)
        if token.casefold() not in _STOP_WORDS and len(token) > 1
    ]
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _scope_clause(scope: str, *, column: str) -> tuple[str, list[str]]:
    """生成精确目录或其子路径的参数化 SQL 条件与绑定参数。"""
    normalized = scope.strip("./")
    if not normalized:
        return "", []
    return f"AND ({column} = ? OR {column} LIKE ?)", [normalized, f"{normalized}/%"]


def _matching_line(content: str, expression: str) -> tuple[int, str]:
    """返回块中首个命中词所在的 0-based 偏移和去空白文本。"""
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
