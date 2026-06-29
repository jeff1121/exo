"""此說明已翻譯為繁體中文。"""

import asyncio
from typing import Generator

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """此說明已翻譯為繁體中文。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_event_loop():
    """此說明已翻譯為繁體中文。"""
    # 已翻譯註解。


def get_pipeline_shard_metadata(
    model_id: ModelId, device_rank: int, world_size: int = 1
) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100000),
            n_layers=32,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=True,  # 已翻譯註解。
    )
    yield caplog
    logger.remove(handler_id)
