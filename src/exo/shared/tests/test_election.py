import pytest
from anyio import create_task_group, fail_after, move_on_after

from exo.routing.connection_message import ConnectionMessage
from exo.shared.election import Election, ElectionMessage, ElectionResult
from exo.shared.types.commands import ForwarderCommand, TestCommand
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.utils.channels import channel

# ======= #
# 已翻譯註解。
# ======= #


def em(
    clock: int,
    seniority: int,
    node_id: str,
    commands_seen: int = 0,
    election_clock: int | None = None,
) -> ElectionMessage:
    """
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    return ElectionMessage(
        clock=clock,
        seniority=seniority,
        proposed_session=SessionId(
            master_node_id=NodeId(node_id),
            election_clock=clock if election_clock is None else election_clock,
        ),
        commands_seen=commands_seen,
    )


# ======================================= #
#                 已翻譯註解。
# ======================================= #


@pytest.fixture(autouse=True)
def fast_election_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("exo.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.1)


@pytest.mark.anyio
async def test_single_round_broadcasts_and_updates_seniority_on_self_win() -> None:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    # 已翻譯註解。
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    # 已翻譯註解。
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    # 已翻譯註解。
    er_tx, er_rx = channel[ElectionResult]()
    # 已翻譯註解。
    cm_tx, cm_rx = channel[ConnectionMessage]()
    # 已翻譯註解。
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)
            # 已翻譯註解。
            await em_in_tx.send(em(clock=1, seniority=0, node_id="A"))

            # 已翻譯註解。
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == NodeId(
                    "B"
                ):
                    break

            # 已翻譯註解。
            result = await er_rx.receive()
            assert result.session_id.master_node_id == NodeId("B")
            # 已翻譯註解。
            assert result.is_new_master is False

            # 已翻譯註解。
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # 已翻譯註解。
    assert election.seniority == 2


@pytest.mark.anyio
async def test_peer_with_higher_seniority_wins_and_we_switch_master() -> None:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # 已翻譯註解。
            await em_in_tx.send(em(clock=1, seniority=10, node_id="PEER"))

            # 已翻譯註解。
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1:
                    assert got.seniority == 0
                    break

            # 已翻譯註解。
            # 已翻譯註解。
            while True:
                result = await er_rx.receive()
                if result.session_id.election_clock == 1:
                    break

            assert result.session_id.master_node_id == NodeId("PEER")
            assert result.is_new_master is True

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # 已翻譯註解。
    assert election.seniority == 0


@pytest.mark.anyio
async def test_ignores_older_messages() -> None:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # 已翻譯註解。
            await em_in_tx.send(em(clock=2, seniority=0, node_id="A"))
            while True:
                first = await em_out_rx.receive()
                if first.clock == 2:
                    break

            # 已翻譯註解。
            await em_in_tx.send(em(clock=1, seniority=999, node_id="B"))

            got_second = False
            with move_on_after(0.05):
                _ = await em_out_rx.receive()
                got_second = True
            assert not got_second, "Should not receive a broadcast for an older round"

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # 已翻譯註解。


@pytest.mark.anyio
async def test_two_rounds_emit_two_broadcasts_and_increment_clock() -> None:
    """
    此說明已翻譯為繁體中文。
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # 已翻譯註解。
            await em_in_tx.send(em(clock=1, seniority=0, node_id="X"))
            while True:
                m1 = await em_out_rx.receive()
                if m1.clock == 1:
                    break

            # 已翻譯註解。
            await em_in_tx.send(em(clock=2, seniority=0, node_id="Y"))
            while True:
                m2 = await em_out_rx.receive()
                if m2.clock == 2:
                    break

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # 已翻譯註解。


@pytest.mark.anyio
async def test_promotion_new_seniority_counts_participants() -> None:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # 已翻譯註解。
            await em_in_tx.send(em(clock=7, seniority=0, node_id="A"))
            await em_in_tx.send(em(clock=7, seniority=0, node_id="B"))

            # 已翻譯註解。
            while True:
                got = await em_out_rx.receive()
                if got.clock == 7 and got.proposed_session.master_node_id == NodeId(
                    "ME"
                ):
                    break

            # 已翻譯註解。
            _ = await er_rx.receive()

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # 已翻譯註解。
    assert election.seniority == 3


@pytest.mark.anyio
async def test_connection_message_triggers_new_round_broadcast() -> None:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # 已翻譯註解。
            await cm_tx.send(ConnectionMessage(connected=True))

            # 已翻譯註解。
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == NodeId(
                    "ME"
                ):
                    break

            # 已翻譯註解。
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # 已翻譯註解。


@pytest.mark.anyio
async def test_tie_breaker_prefers_node_with_more_commands_seen() -> None:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    me = NodeId("ME")

    election = Election(
        node_id=me,
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
        seniority=0,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # 已翻譯註解。
            for _ in range(50):
                await co_tx.send(
                    ForwarderCommand(origin=SystemId("SOMEONE"), command=TestCommand())
                )

            # 已翻譯註解。
            await em_in_tx.send(
                em(clock=1, seniority=0, node_id="PEER", commands_seen=5)
            )

            # 已翻譯註解。
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == me:
                    # 已翻譯註解。
                    break

            # 已翻譯註解。
            while True:
                result = await er_rx.receive()
                if result.session_id.master_node_id == me:
                    assert result.session_id.election_clock in (0, 1)
                    break

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()
