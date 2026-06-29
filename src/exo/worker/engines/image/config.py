from enum import Enum

from pydantic import BaseModel


class BlockType(Enum):
    JOINT = "joint"  # 已翻譯註解。
    SINGLE = "single"  # 已翻譯註解。


class TransformerBlockConfig(BaseModel):
    model_config = {"frozen": True}

    block_type: BlockType
    count: int
    has_separate_text_output: bool  # 已翻譯註解。


class ImageModelConfig(BaseModel):
    model_family: str

    block_configs: tuple[TransformerBlockConfig, ...]

    default_steps: dict[str, int]  # 已翻譯註解。
    num_sync_steps: int  # 已翻譯註解。

    guidance_scale: float | None = None  # 已翻譯註解。

    @property
    def total_blocks(self) -> int:
        return sum(bc.count for bc in self.block_configs)

    @property
    def joint_block_count(self) -> int:
        return sum(
            bc.count for bc in self.block_configs if bc.block_type == BlockType.JOINT
        )

    @property
    def single_block_count(self) -> int:
        return sum(
            bc.count for bc in self.block_configs if bc.block_type == BlockType.SINGLE
        )

    def get_steps_for_quality(self, quality: str) -> int:
        return self.default_steps[quality]
