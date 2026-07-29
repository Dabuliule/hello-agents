from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace ``path`` with fully written text.

    The temporary file lives beside the destination so ``os.replace`` stays on the
    same filesystem. Existing permission bits are preserved; a newly created file
    keeps the restrictive permissions chosen by ``mkstemp``.
    """
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True

    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as stream:
            descriptor_open = False
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        if path.exists():
            shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        if descriptor_open:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
