import exo.worker.plan as plan_mod
from exo.shared.types.tasks import StartWarmup
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerWarmingUp,
)
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    NODE_C,
    RUNNER_1_ID,
    RUNNER_2_ID,
    RUNNER_3_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def test_plan_starts_warmup_for_accepting_rank_when_all_loaded_or_warming():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=3)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=3)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=2, world_size=3)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID, NODE_C: RUNNER_3_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1, RUNNER_3_ID: shard2},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_2_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerLoaded(),
        RUNNER_3_ID: RunnerWarmingUp(),
    }

    result = plan_mod.plan(
        node_id=NODE_B,
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

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_starts_warmup_for_rank_zero_after_others_warming():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerWarmingUp(),
    }

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

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_start_warmup_for_non_zero_rank_until_all_loaded_or_warming():
    """
    此說明已翻譯為繁體中文。
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_2_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerIdle(),
        RUNNER_2_ID: RunnerLoaded(),
    }

    result = plan_mod.plan(
        node_id=NODE_B,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None


def test_plan_does_not_start_warmup_for_rank_zero_until_others_warming():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    # 已翻譯註解。
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerLoaded(),
    }

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

    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerWarmingUp(),
    }

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

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_starts_warmup_for_connecting_rank_after_others_warming():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    # 已翻譯註解。
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_2_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerWarmingUp(),
        RUNNER_2_ID: RunnerLoaded(),
    }

    result = plan_mod.plan(
        node_id=NODE_B,
        runners=runners,  # type: ignore
        global_download_status={NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_start_warmup_for_accepting_rank_until_all_loaded_or_warming():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    # 已翻譯註解。
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerLoading(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None


def test_plan_does_not_start_warmup_for_connecting_rank_until_others_warming():
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )

    # 已翻譯註解。
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerLoaded(),
        RUNNER_2_ID: RunnerLoaded(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None
