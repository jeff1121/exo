import contextlib
import json
from collections import OrderedDict
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BufferedRandom, BufferedReader
from pathlib import Path

import msgspec
import zstandard
from loguru import logger
from pydantic import TypeAdapter

from exo.shared.types.events import Event

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)

_HEADER_SIZE = 4  # 已翻譯註解。
_OFFSET_CACHE_SIZE = 128
_MAX_ARCHIVES = 5


def _serialize_event(event: Event) -> bytes:
    return msgspec.msgpack.encode(event.model_dump(mode="json"))


def _deserialize_event(raw: bytes) -> Event:
    # 已翻譯註解。
    # 已翻譯註解。
    # 已翻譯註解。
    # 已翻譯註解。
    # 取得正確往返反序列化的唯一方式。
    as_json = json.dumps(msgspec.msgpack.decode(raw, type=dict))
    return _EVENT_ADAPTER.validate_json(as_json)


def _unpack_header(header: bytes) -> int:
    return int.from_bytes(header, byteorder="big")


def _skip_record(f: BufferedReader) -> bool:
    """此說明已翻譯為繁體中文。"""
    header = f.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        return False
    f.seek(_unpack_header(header), 1)
    return True


def _read_record(f: BufferedReader) -> Event | None:
    """此說明已翻譯為繁體中文。"""
    header = f.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        return None
    length = _unpack_header(header)
    payload = f.read(length)
    if len(payload) < length:
        return None
    return _deserialize_event(payload)


class DiskEventLog:
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._active_path = directory / "events.bin"
        self._offset_cache: OrderedDict[int, int] = OrderedDict()
        self._count: int = 0

        # 已翻譯註解。
        if self._active_path.exists():
            self._rotate(self._active_path, self._directory)

        self._file: BufferedRandom = open(self._active_path, "w+b")  # noqa: SIM115

    def _cache_offset(self, idx: int, offset: int) -> None:
        self._offset_cache[idx] = offset
        self._offset_cache.move_to_end(idx)
        if len(self._offset_cache) > _OFFSET_CACHE_SIZE:
            self._offset_cache.popitem(last=False)

    def _seek_to(self, f: BufferedReader, target_idx: int) -> None:
        """此說明已翻譯為繁體中文。"""
        if target_idx in self._offset_cache:
            self._offset_cache.move_to_end(target_idx)
            f.seek(self._offset_cache[target_idx])
            return

        # 已翻譯註解。
        scan_from_idx = 0
        scan_from_offset = 0
        for cached_idx in self._offset_cache:
            if cached_idx < target_idx:
                scan_from_idx = cached_idx
                scan_from_offset = self._offset_cache[cached_idx]

        # 向前掃描並略過紀錄
        f.seek(scan_from_offset)
        for _ in range(scan_from_idx, target_idx):
            _skip_record(f)

        self._cache_offset(target_idx, f.tell())

    def append(self, event: Event) -> None:
        packed = _serialize_event(event)
        self._file.write(len(packed).to_bytes(_HEADER_SIZE, byteorder="big"))
        self._file.write(packed)
        self._count += 1

    def read_range(self, start: int, end: int) -> Iterator[Event]:
        """此說明已翻譯為繁體中文。"""
        end = min(end, self._count)
        if start < 0 or end < 0 or start >= end:
            return

        self._file.flush()
        with open(self._active_path, "rb") as f:
            self._seek_to(f, start)
            for _ in range(end - start):
                event = _read_record(f)
                if event is None:
                    break
                yield event

            # 快取目前結束位置，讓下一次連續讀取可命中
            if end < self._count:
                self._cache_offset(end, f.tell())

    def read_all(self) -> Iterator[Event]:
        """逐筆產生日誌中的所有事件。"""
        if self._count == 0:
            return
        self._file.flush()
        with open(self._active_path, "rb") as f:
            for _ in range(self._count):
                event = _read_record(f)
                if event is None:
                    break
                yield event

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        """此說明已翻譯為繁體中文。"""
        if self._file.closed:
            return
        self._file.close()
        if self._active_path.exists() and self._count > 0:
            self._rotate(self._active_path, self._directory)
        elif self._active_path.exists():
            self._active_path.unlink()

    @staticmethod
    def _rotate(source: Path, directory: Path) -> None:
        """將來源檔壓縮為含時間戳的封存檔。

        此說明已翻譯為繁體中文。
        """
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
            dest = directory / f"events.{stamp}.bin.zst"
            compressor = zstandard.ZstdCompressor()
            with open(source, "rb") as f_in, open(dest, "wb") as f_out:
                compressor.copy_stream(f_in, f_out)
            source.unlink()
            logger.info(f"Rotated event log: {source} -> {dest}")

            # 清理超過上限的最舊封存檔
            archives = sorted(directory.glob("events.*.bin.zst"))
            for old in archives[:-_MAX_ARCHIVES]:
                old.unlink()
        except Exception as e:
            logger.opt(exception=e).warning(f"Failed to rotate event log {source}")
            # 即使壓縮失敗也嘗試清理來源檔
            with contextlib.suppress(OSError):
                source.unlink()
