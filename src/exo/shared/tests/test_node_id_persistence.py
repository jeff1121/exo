import contextlib
import multiprocessing
import os
from multiprocessing import Event, Queue, Semaphore
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as QueueT
from multiprocessing.synchronize import Event as EventT
from multiprocessing.synchronize import Semaphore as SemaphoreT

from loguru import logger
from pytest import LogCaptureFixture, mark

from exo.routing.router import get_node_zid
from exo.shared.constants import EXO_NODE_ZID

NUM_CONCURRENT_PROCS = 10


def _get_keypair_concurrent_subprocess_task(
    sem: SemaphoreT, ev: EventT, queue: QueueT[bytes]
) -> None:
    # 已翻譯註解。
    sem.release()
    # 已翻譯註解。
    ev.wait()
    queue.put(get_node_zid().encode())


def _get_keypair_concurrent(num_procs: int) -> bytes:
    assert num_procs > 0

    sem = Semaphore(0)
    ev = Event()
    queue: QueueT[bytes] = Queue(maxsize=num_procs)

    # 已翻譯註解。
    logger.info(f"PARENT: Starting {num_procs} subprocesses")
    ps: list[BaseProcess] = []
    for _ in range(num_procs):
        p = multiprocessing.get_context("fork").Process(
            target=_get_keypair_concurrent_subprocess_task, args=(sem, ev, queue)
        )
        ps.append(p)
        p.start()
    for _ in range(num_procs):
        sem.acquire()

    # 已翻譯註解。
    logger.info("PARENT: Beginning read")
    ev.set()

    # 已翻譯註解。
    for p in ps:
        p.join()

    # 已翻譯註解。
    # 已翻譯註解。
    logger.info("PARENT: Checking consistency")
    keypair: bytes | None = None
    qsize = 0  # 已翻譯註解。
    while not queue.empty():
        qsize += 1
        temp_keypair = queue.get()
        if keypair is None:
            keypair = temp_keypair
        else:
            assert keypair == temp_keypair
    assert num_procs == qsize
    return keypair  # 已翻譯註解。


def _delete_if_exists(p: str | bytes | os.PathLike[str] | os.PathLike[bytes]):
    with contextlib.suppress(OSError):
        os.remove(p)


@mark.skip(reason="this functionality is currently disabled but may return in future")
def test_node_id_fetching(caplog: LogCaptureFixture):
    reps = 10

    # 已翻譯註解。
    _delete_if_exists(EXO_NODE_ZID)
    kp = _get_keypair_concurrent(NUM_CONCURRENT_PROCS)

    with caplog.at_level(101):  # 已翻譯註解。
        # 已翻譯註解。
        for _ in range(reps):
            assert kp == _get_keypair_concurrent(NUM_CONCURRENT_PROCS)

        # 已翻譯註解。
        _delete_if_exists(EXO_NODE_ZID)
        for _ in range(reps):
            assert kp != _get_keypair_concurrent(NUM_CONCURRENT_PROCS)
