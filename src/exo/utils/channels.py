import contextlib
import multiprocessing as mp
from dataclasses import dataclass, field
from functools import wraps
from inspect import iscoroutinefunction
from math import inf
from multiprocessing.synchronize import Event
from queue import Empty, Full
from types import CoroutineType, TracebackType
from typing import Any, Callable, NoReturn, Self, cast, overload, override

from anyio import (
    BrokenResourceError,
    CapacityLimiter,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    to_thread,
)
from anyio.streams.memory import (
    MemoryObjectReceiveStream as AnyioReceiver,
)
from anyio.streams.memory import (
    MemoryObjectSendStream as AnyioSender,
)
from anyio.streams.memory import (
    MemoryObjectStreamState,
)
from anyio.streams.memory import (
    MemoryObjectStreamState as AnyioState,
)


@dataclass(eq=False)
class ErrorOverride:
    closed_resource_error: type[ClosedResourceError] = field(
        default=ClosedResourceError,
    )
    broken_resource_error: type[BrokenResourceError] = field(
        default=BrokenResourceError,
    )
    end_of_stream: type[EndOfStream] = field(
        default=EndOfStream,
    )
    would_block: type[WouldBlock] = field(
        default=WouldBlock,
    )

    @overload
    def patch[**P, R](
        self,
        fn: Callable[P, CoroutineType[Any, Any, R]],
        /,
    ) -> Callable[P, CoroutineType[Any, Any, R]]: ...

    @overload
    def patch[**P, R](
        self,
        fn: Callable[P, R],
        /,
    ) -> Callable[P, R]: ...

    def patch[**P, R](self, fn: Callable[P, Any], /) -> Callable[P, Any]:
        """回傳一個函式，將這些例外替換為對應的覆寫例外。"""

        if iscoroutinefunction(fn):
            async_fn = cast(Callable[P, CoroutineType[Any, Any, R]], fn)

            @wraps(async_fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    return await async_fn(*args, **kwargs)
                except ClosedResourceError as e:
                    self._raise_replace(self.closed_resource_error, e)
                except BrokenResourceError as e:
                    self._raise_replace(self.broken_resource_error, e)
                except EndOfStream as e:
                    self._raise_replace(self.end_of_stream, e)
                except WouldBlock as e:
                    self._raise_replace(self.would_block, e)

            return async_wrapper
        else:
            sync_fn = cast(Callable[P, R], fn)

            @wraps(sync_fn)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    return sync_fn(*args, **kwargs)
                except ClosedResourceError as e:
                    self._raise_replace(self.closed_resource_error, e)
                except BrokenResourceError as e:
                    self._raise_replace(self.broken_resource_error, e)
                except EndOfStream as e:
                    self._raise_replace(self.end_of_stream, e)
                except WouldBlock as e:
                    self._raise_replace(self.would_block, e)

            return sync_wrapper

    @staticmethod
    def _raise_replace(replacement: type[BaseException], e: BaseException) -> NoReturn:
        if isinstance(e, replacement):
            raise
        raise replacement() from e


class Sender[T](AnyioSender[T]):
    def __init__(
        self,
        state: MemoryObjectStreamState[T],
        error_override_config: ErrorOverride | None,
    ):
        super().__init__(_state=state)

        # 替要覆寫錯誤型別的方法打補丁
        #
        # 注意：若新增的方法可能拋出例外，務必在此區塊補丁
        if (e := error_override_config) is not None:
            # 本類別新增的方法
            self.clone_receiver = e.patch(self.clone_receiver)

            # 覆寫的方法
            self.clone = e.patch(self.clone)

            # 父類別方法
            self.send_nowait = e.patch(self.send_nowait)
            self.send = e.patch(self.send)
            self.close = e.patch(self.close)
            self.aclose = e.patch(self.aclose)
            self.statistics = e.patch(self.statistics)

        self.err_config = error_override_config

    @override
    def clone(self) -> "Sender[T]":
        if self._closed:
            raise ClosedResourceError
        return Sender(self._state, self.err_config)

    def clone_receiver(self) -> "Receiver[T]":
        """此說明已翻譯為繁體中文。"""
        if self._closed:
            raise ClosedResourceError
        return Receiver(self._state, self.err_config)


class Receiver[T](AnyioReceiver[T]):
    def __init__(
        self,
        state: MemoryObjectStreamState[T],
        error_override_config: ErrorOverride | None,
    ):
        super().__init__(_state=state)

        # 替要覆寫錯誤型別的方法打補丁
        #
        # 注意：若新增的方法可能拋出例外，務必在此區塊補丁
        if (e := error_override_config) is not None:
            # 本類別新增的方法
            self.clone_sender = e.patch(self.clone_sender)
            self.collect = e.patch(self.collect)
            self.receive_at_least = e.patch(self.receive_at_least)

            # 覆寫的方法
            self.clone = e.patch(self.clone)

            # 父類別方法
            self.receive_nowait = e.patch(self.receive_nowait)
            self.receive = e.patch(self.receive)
            self.close = e.patch(self.close)
            self.aclose = e.patch(self.aclose)
            self.statistics = e.patch(self.statistics)

        self.err_config = error_override_config

    @override
    def clone(self) -> "Receiver[T]":
        if self._closed:
            raise ClosedResourceError
        return Receiver(self._state, self.err_config)

    def clone_sender(self) -> Sender[T]:
        """此說明已翻譯為繁體中文。"""
        if self._closed:
            raise ClosedResourceError
        return Sender(self._state, self.err_config)

    def collect(self) -> list[T]:
        """此說明已翻譯為繁體中文。"""
        out: list[T] = []
        while True:
            try:
                item = self.receive_nowait()
                out.append(item)
            except WouldBlock:
                break
        return out

    async def receive_at_least(self, n: int) -> list[T]:
        out: list[T] = []
        out.append(await self.receive())
        out.extend(self.collect())
        while len(out) < n:
            out.append(await self.receive())
            out.extend(self.collect())
        return out

    @override
    def __enter__(self) -> Self:
        return self


class _MpEndOfStream:
    pass


class MpState[T]:
    def __init__(self, max_buffer_size: float):
        if max_buffer_size == inf:
            max_buffer_size = 0
        assert isinstance(max_buffer_size, int), (
            "State should only ever be constructed with an integer or math.inf size."
        )

        self.max_buffer_size: float = max_buffer_size
        self.buffer: mp.Queue[T | _MpEndOfStream] = mp.Queue(max_buffer_size)
        self.closed: Event = mp.Event()

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


@dataclass(eq=False)
class MpSender[T]:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    _state: MpState[T] = field()

    def send_nowait(self, item: T) -> None:
        if self._state.closed.is_set():
            raise ClosedResourceError
        try:
            self._state.buffer.put(item, block=False)
        except Full:
            raise WouldBlock from None
        except ValueError as e:
            print("Unreachable code path - let me know!")
            raise ClosedResourceError from e

    def send(self, item: T) -> None:
        if self._state.closed.is_set():
            raise ClosedResourceError
        try:
            self.send_nowait(item)
        except WouldBlock:
            # 已翻譯註解。
            self._state.buffer.put(item, block=True)

    async def send_async(self, item: T) -> None:
        await to_thread.run_sync(
            self.send, item, limiter=CapacityLimiter(1), abandon_on_cancel=True
        )

    def close(self) -> None:
        if not self._state.closed.is_set():
            self._state.closed.set()
        with contextlib.suppress(Exception):
            self._state.buffer.put_nowait(_MpEndOfStream())
        self._state.buffer.close()

    # 已翻譯註解。
    def join(self) -> None:
        """確保佇列中的訊息都處理完成後再繼續。"""
        assert self._state.closed.is_set(), (
            "Mp channels must be closed before being joined"
        )
        self._state.buffer.join_thread()

    # 已翻譯註解。
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


@dataclass(eq=False)
class MpReceiver[T]:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    _state: MpState[T] = field()

    def receive_nowait(self) -> T:
        if self._state.closed.is_set():
            raise ClosedResourceError

        try:
            item = self._state.buffer.get(block=False)
            if isinstance(item, _MpEndOfStream):
                self.close()
                raise EndOfStream
            return item
        except Empty:
            raise WouldBlock from None
        except ValueError as e:
            print("Unreachable code path - let me know!")
            raise ClosedResourceError from e

    def receive(self) -> T:
        try:
            return self.receive_nowait()
        except WouldBlock:
            try:
                item = self._state.buffer.get()
            except (TypeError, OSError):
                # 已翻譯註解。
                # 已翻譯註解。
                # 已翻譯註解。
                raise ClosedResourceError from None
            if isinstance(item, _MpEndOfStream):
                self.close()
                raise EndOfStream from None
            return item

    async def receive_async(self) -> T:
        return await to_thread.run_sync(
            self.receive, limiter=CapacityLimiter(1), abandon_on_cancel=True
        )

    def close(self) -> None:
        if not self._state.closed.is_set():
            self._state.closed.set()
        with contextlib.suppress(Exception):
            self._state.buffer.put_nowait(_MpEndOfStream())
        self._state.buffer.close()

    # 已翻譯註解。
    def join(self) -> None:
        """阻塞直到我們這端緩衝區中排隊訊息都被清空。"""
        assert self._state.closed.is_set(), (
            "Mp channels must be closed before being joined"
        )
        self._state.buffer.join_thread()

    # == 迭代器支援 ==
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        try:
            return self.receive()
        except EndOfStream:
            raise StopIteration from None

    # == 非同步迭代器支援 ==
    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.receive_async()
        except EndOfStream:
            raise StopAsyncIteration from None

    # 已翻譯註解。
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def collect(self) -> list[T]:
        """此說明已翻譯為繁體中文。"""
        out: list[T] = []
        while True:
            try:
                item = self.receive_nowait()
                out.append(item)
            except WouldBlock:
                break
        return out

    def receive_at_least(self, n: int) -> list[T]:
        out: list[T] = []
        out.append(self.receive())
        out.extend(self.collect())
        while len(out) < n:
            out.append(self.receive())
            out.extend(self.collect())
        return out

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


class channel[T]:  # noqa: N801
    """建立一對用於同一行程內通訊的非同步通道。"""

    def __new__(
        cls,
        max_buffer_size: float = inf,
        error_override_config: ErrorOverride | None = None,
    ) -> tuple[Sender[T], Receiver[T]]:
        if max_buffer_size != inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")
        state = AnyioState[T](max_buffer_size)
        return Sender(state, error_override_config), Receiver(
            state, error_override_config
        )


class mp_channel[T]:  # noqa: N801
    """建立一對用於跨行程通訊的同步通道。"""

    # 已翻譯註解。
    def __new__(cls, max_buffer_size: float = inf) -> tuple[MpSender[T], MpReceiver[T]]:
        if (
            max_buffer_size == 0
            or max_buffer_size != inf
            and not isinstance(max_buffer_size, int)
        ):
            raise ValueError(
                "max_buffer_size must be either an integer or math.inf. 0-sized buffers are not supported by multiprocessing"
            )
        state = MpState[T](max_buffer_size)
        return MpSender(_state=state), MpReceiver(_state=state)
