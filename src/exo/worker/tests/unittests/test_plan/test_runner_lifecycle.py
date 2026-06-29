from typing import Any

import exo.worker.plan as plan_mod
from exo.shared.types.tasks import Shutdown
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerStatus,
)
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def test_plan_kills_runner_when_instance_missing():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances: dict[InstanceId, Instance] = {}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_kills_runner_when_sibling_failed():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerFailed(error_message="boom", diagnostics=[]),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_creates_runner_when_missing_for_node():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )

    runners: dict[Any, Any] = {}  # 已翻譯註解。
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[Any, Any] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    # 已翻譯註解。
    assert isinstance(result, plan_mod.CreateRunner)
    assert result.instance_id == INSTANCE_1_ID
    assert isinstance(result.bound_instance, BoundInstance)
    assert result.bound_instance.instance is instance
    assert result.bound_instance.bound_runner_id == RUNNER_1_ID


def test_plan_does_not_create_runner_when_supervisor_already_present():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None


def test_plan_does_not_create_runner_for_unassigned_node():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_2_ID: shard},
    )

    runners: dict[RunnerId, FakeRunnerSupervisor] = {}  # 已翻譯註解。
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[RunnerId, RunnerStatus] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None
