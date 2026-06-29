from loguru import logger


class OrderedBuffer[T]:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    def __init__(self):
        self.store: dict[int, T] = {}
        self.next_idx_to_release: int = 0

    def ingest(self, idx: int, t: T):
        """此說明已翻譯為繁體中文。"""
        logger.trace(f"Ingested event {t}")
        if idx < self.next_idx_to_release:
            return
        if idx in self.store:
            assert self.store[idx] == t, (
                "Received different messages with identical indices, probable race condition"
            )
            return
        self.store[idx] = t

    def drain(self) -> list[T]:
        """此說明已翻譯為繁體中文。"""
        ret: list[T] = []
        while self.next_idx_to_release in self.store:
            idx = self.next_idx_to_release
            event = self.store.pop(idx)
            ret.append(event)
            self.next_idx_to_release += 1
        logger.trace(f"Releasing event {ret}")
        return ret

    def drain_indexed(self) -> list[tuple[int, T]]:
        """此說明已翻譯為繁體中文。"""
        ret: list[tuple[int, T]] = []
        while self.next_idx_to_release in self.store:
            idx = self.next_idx_to_release
            event = self.store.pop(idx)
            ret.append((idx, event))
            self.next_idx_to_release += 1
        logger.trace(f"Releasing event {ret}")
        return ret


class MultiSourceBuffer[SourceId, T]:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    def __init__(self):
        self.stores: dict[SourceId, OrderedBuffer[T]] = {}

    def ingest(self, idx: int, t: T, source: SourceId):
        if source not in self.stores:
            self.stores[source] = OrderedBuffer()
        buffer = self.stores[source]
        buffer.ingest(idx, t)

    def drain(self) -> list[T]:
        ret: list[T] = []
        for store in self.stores.values():
            ret.extend(store.drain())
        return ret
