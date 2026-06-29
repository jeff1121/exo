from enum import Enum
from typing import TypeAlias, final

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"


class BaseShardMetadata(TaggedModel):
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    model_card: ModelCard
    device_rank: int
    world_size: int

    # 已翻譯註解。
    # 已翻譯註解。
    immediate_exception: bool = False
    should_timeout: float | None = None

    start_layer: int = Field(ge=0)
    end_layer: int = Field(ge=0)
    n_layers: int = Field(ge=0)

    @property
    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers

    def __hash__(self) -> int:
        return hash(
            (
                self.model_card.model_id,
                self.start_layer,
                self.end_layer,
                self.n_layers,
                self.device_rank,
                self.world_size,
            )
        )


@final
class PipelineShardMetadata(BaseShardMetadata):
    """
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """


@final
class CfgShardMetadata(BaseShardMetadata):
    """此說明已翻譯為繁體中文。"""

    cfg_rank: int  # 已翻譯註解。
    cfg_world_size: int = 2

    # 已翻譯註解。
    pipeline_rank: int  # 已翻譯註解。
    pipeline_world_size: int  # 已翻譯註解。


@final
class TensorShardMetadata(BaseShardMetadata):
    pass


ShardMetadata: TypeAlias = (
    PipelineShardMetadata | CfgShardMetadata | TensorShardMetadata
)
