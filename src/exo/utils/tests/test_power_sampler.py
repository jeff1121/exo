from collections.abc import Mapping

import anyio
import pytest

from exo.api.types import PowerUsage
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import SystemPerformanceProfile
from exo.utils.power_sampler import PowerSampler


def _make_profile(sys_power: float) -> SystemPerformanceProfile:
    return SystemPerformanceProfile(sys_power=sys_power)


NODE_A = NodeId("node-a")
NODE_B = NodeId("node-b")


@pytest.fixture
def single_node_sampler() -> PowerSampler:
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    return PowerSampler(get_node_system=lambda: state)


@pytest.fixture
def multi_node_state() -> dict[NodeId, SystemPerformanceProfile]:
    return {
        NODE_A: _make_profile(10.0),
        NODE_B: _make_profile(20.0),
    }


async def test_single_sample(single_node_sampler: PowerSampler) -> None:
    """此說明已翻譯為繁體中文。"""
    async with anyio.create_task_group() as tg:
        tg.start_soon(single_node_sampler.run)
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    result = single_node_sampler.result()
    assert len(result.nodes) == 1
    assert result.nodes[0].node_id == NODE_A
    assert result.nodes[0].avg_sys_power == 10.0
    assert result.nodes[0].samples >= 1
    assert result.elapsed_seconds > 0


async def test_multi_node_averaging(
    multi_node_state: dict[NodeId, SystemPerformanceProfile],
) -> None:
    """此說明已翻譯為繁體中文。"""
    sampler = PowerSampler(get_node_system=lambda: multi_node_state)
    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    result = sampler.result()
    assert len(result.nodes) == 2
    assert result.total_avg_sys_power_watts == 30.0


async def test_energy_calculation(single_node_sampler: PowerSampler) -> None:
    """此說明已翻譯為繁體中文。"""
    async with anyio.create_task_group() as tg:
        tg.start_soon(single_node_sampler.run)
        await anyio.sleep(0.3)
        tg.cancel_scope.cancel()

    result = single_node_sampler.result()
    expected_energy = result.total_avg_sys_power_watts * result.elapsed_seconds
    assert result.total_energy_joules == expected_energy


async def test_changing_power_is_averaged() -> None:
    """此說明已翻譯為繁體中文。"""
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    sampler = PowerSampler(get_node_system=lambda: state, interval=0.05)

    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.15)
        state[NODE_A] = _make_profile(20.0)
        await anyio.sleep(0.15)
        tg.cancel_scope.cancel()

    result = sampler.result()
    avg = result.nodes[0].avg_sys_power
    # 已翻譯註解。
    assert 10.0 < avg < 20.0


async def test_empty_state() -> None:
    """此說明已翻譯為繁體中文。"""
    empty: Mapping[NodeId, SystemPerformanceProfile] = {}
    sampler = PowerSampler(get_node_system=lambda: empty)

    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    result = sampler.result()
    assert len(result.nodes) == 0
    assert result.total_avg_sys_power_watts == 0.0
    assert result.total_energy_joules == 0.0


def test_trapezoidal_unit_dt_weighting() -> None:
    """此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。"""
    from exo.utils.power_sampler import trapezoidal_energy

    # 已翻譯註解。
    # 已翻譯註解。
    samples = [
        (0.0, _make_profile(10.0)),
        (4.9, _make_profile(10.0)),
        (5.0, _make_profile(100.0)),
    ]
    energy = trapezoidal_energy(samples, elapsed=5.0)
    # 已翻譯註解。
    assert abs(energy - 54.5) < 1e-9
    avg = energy / 5.0  # 已翻譯註解。
    # 已翻譯註解。
    # 已翻譯註解。
    assert abs(avg - 10.9) < 1e-9


def test_trapezoidal_unit_single_sample() -> None:
    """此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。"""
    from exo.utils.power_sampler import trapezoidal_energy

    samples = [(0.0, _make_profile(42.0))]
    assert trapezoidal_energy(samples, elapsed=3.0) == 42.0 * 3.0


def test_trapezoidal_range_interpolation() -> None:
    """此說明已翻譯為繁體中文。"""
    from exo.utils.power_sampler import trapezoidal_energy_range

    # 已翻譯註解。
    samples = [
        (0.0, _make_profile(10.0)),
        (10.0, _make_profile(20.0)),
    ]
    # 已翻譯註解。
    assert abs(trapezoidal_energy_range(samples, 4.0, 6.0) - 30.0) < 1e-9
    # 已翻譯註解。
    full = trapezoidal_energy_range(samples, 0.0, 10.0)
    assert abs(full - 150.0) < 1e-9


def test_trapezoidal_range_zero_window() -> None:
    """此說明已翻譯為繁體中文。"""
    from exo.utils.power_sampler import trapezoidal_energy_range

    samples = [(0.0, _make_profile(10.0)), (5.0, _make_profile(20.0))]
    assert trapezoidal_energy_range(samples, 3.0, 3.0) == 0.0
    assert trapezoidal_energy_range(samples, 5.0, 3.0) == 0.0


def test_trapezoidal_range_splits_sum_to_full() -> None:
    """此說明已翻譯為繁體中文。"""
    from exo.utils.power_sampler import (
        trapezoidal_energy,
        trapezoidal_energy_range,
    )

    samples = [
        (0.0, _make_profile(10.0)),
        (1.0, _make_profile(20.0)),
        (3.0, _make_profile(15.0)),
        (5.0, _make_profile(25.0)),
    ]
    full = trapezoidal_energy(samples, elapsed=5.0)
    # 已翻譯註解。
    left = trapezoidal_energy_range(samples, 0.0, 2.5)
    right = trapezoidal_energy_range(samples, 2.5, 5.0)
    assert abs((left + right) - full) < 1e-9


async def test_prefill_generation_split() -> None:
    """此說明已翻譯為繁體中文。"""
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    sampler = PowerSampler(get_node_system=lambda: state, interval=0.02)

    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        # 已翻譯註解。
        await anyio.sleep(0.1)
        # 已翻譯註解。
        # 已翻譯註解。
        # 已翻譯註解。
        # 已翻譯註解。
        sampler.mark_prefill_done()
        state[NODE_A] = _make_profile(30.0)
        # 已翻譯註解。
        await anyio.sleep(0.1)
        tg.cancel_scope.cancel()

    result = sampler.result()
    assert result.prefill_seconds is not None
    assert result.generation_seconds is not None
    assert result.prefill_energy_joules is not None
    assert result.generation_energy_joules is not None
    assert result.prefill_avg_sys_power_watts is not None
    assert result.generation_avg_sys_power_watts is not None

    # 已翻譯註解。
    assert (
        abs(
            (result.prefill_seconds + result.generation_seconds)
            - result.elapsed_seconds
        )
        < 1e-6
    )
    # 已翻譯註解。
    assert (
        abs(
            (result.prefill_energy_joules + result.generation_energy_joules)
            - result.total_energy_joules
        )
        < 1e-6
    )
    # 已翻譯註解。
    # 已翻譯註解。
    # 已翻譯註解。
    assert result.prefill_avg_sys_power_watts < 15.0
    assert result.generation_avg_sys_power_watts > 15.0
    assert result.nodes[0].prefill_avg_sys_power is not None
    assert result.nodes[0].generation_avg_sys_power is not None


async def test_no_split_when_unmarked() -> None:
    """此說明已翻譯為繁體中文。"""
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    sampler = PowerSampler(get_node_system=lambda: state, interval=0.02)
    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    result = sampler.result()
    assert result.prefill_seconds is None
    assert result.generation_seconds is None
    assert result.prefill_energy_joules is None
    assert result.generation_energy_joules is None
    assert result.nodes[0].prefill_energy_joules is None
    assert result.nodes[0].generation_energy_joules is None


async def test_mark_prefill_done_is_idempotent() -> None:
    """此說明已翻譯為繁體中文。"""
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    sampler = PowerSampler(get_node_system=lambda: state, interval=0.02)
    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.05)
        sampler.mark_prefill_done()
        first_prefill_at = sampler._prefill_done_at  # 已翻譯註解。
        await anyio.sleep(0.05)
        sampler.mark_prefill_done()
        assert sampler._prefill_done_at == first_prefill_at  # 已翻譯註解。
        tg.cancel_scope.cancel()


async def test_result_stops_sampling() -> None:
    """此說明已翻譯為繁體中文。"""
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    sampler = PowerSampler(get_node_system=lambda: state, interval=0.02)

    result: PowerUsage | None = None
    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.1)
        result = sampler.result()
        # 已翻譯註解。
        await anyio.sleep(0.1)
        tg.cancel_scope.cancel()

    assert result is not None
    assert result.nodes[0].samples >= 2
