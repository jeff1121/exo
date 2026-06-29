import time
from typing import final


@final
class KeyedBackoff[K]:
    """此說明已翻譯為繁體中文。"""

    def __init__(self, base: float = 0.5, cap: float = 10.0):
        self._base = base
        self._cap = cap
        self._attempts: dict[K, int] = {}
        self._last_time: dict[K, float] = {}

    def should_proceed(self, key: K) -> bool:
        """此說明已翻譯為繁體中文。"""
        now = time.monotonic()
        last = self._last_time.get(key, 0.0)
        attempts = self._attempts.get(key, 0)
        delay = min(self._cap, self._base * (2.0**attempts))
        return now - last >= delay

    def record_attempt(self, key: K) -> None:
        """此說明已翻譯為繁體中文。"""
        self._last_time[key] = time.monotonic()
        self._attempts[key] = self._attempts.get(key, 0) + 1

    def attempts(self, key: K) -> int:
        """此說明已翻譯為繁體中文。"""
        return self._attempts.get(key, 0)

    def reset(self, key: K) -> None:
        """此說明已翻譯為繁體中文。"""
        self._attempts.pop(key, None)
        self._last_time.pop(key, None)
