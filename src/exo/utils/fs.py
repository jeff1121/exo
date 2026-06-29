import contextlib
import os
import pathlib
import tempfile
from typing import LiteralString

type StrPath = str | os.PathLike[str]
type BytesPath = bytes | os.PathLike[bytes]
type StrOrBytesPath = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def delete_if_exists(filename: StrOrBytesPath) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


def ensure_parent_directory_exists(filename: StrPath) -> None:
    """
    此說明已翻譯為繁體中文。
    """
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)


def ensure_directory_exists(dirname: StrPath) -> None:
    """
    此說明已翻譯為繁體中文。
    """
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)


def make_temp_path(name: LiteralString) -> str:
    return os.path.join(tempfile.mkdtemp(), name)
