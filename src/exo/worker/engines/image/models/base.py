from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.utils.image_util import ImageUtil
from PIL import Image

from exo.worker.engines.image.config import ImageModelConfig

if TYPE_CHECKING:
    from exo.worker.engines.image.pipeline.block_wrapper import (
        JointBlockWrapper,
        SingleBlockWrapper,
    )

ModelT = TypeVar("ModelT")
TransformerT = TypeVar("TransformerT")

RotaryEmbeddings = mx.array | tuple[mx.array, mx.array]


class PromptData(ABC):
    @property
    @abstractmethod
    def prompt_embeds(self) -> mx.array: ...

    @property
    @abstractmethod
    def pooled_prompt_embeds(self) -> mx.array: ...

    @property
    @abstractmethod
    def negative_prompt_embeds(self) -> mx.array | None: ...

    @property
    @abstractmethod
    def negative_pooled_prompt_embeds(self) -> mx.array | None: ...

    @abstractmethod
    def get_encoder_hidden_states_mask(
        self, positive: bool = True
    ) -> mx.array | None: ...

    @property
    @abstractmethod
    def cond_image_grid(
        self,
    ) -> tuple[int, int, int] | list[tuple[int, int, int]] | None:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...

    @property
    @abstractmethod
    def conditioning_latents(self) -> mx.array | None:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...

    @property
    @abstractmethod
    def kontext_image_ids(self) -> mx.array | None:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...

    @abstractmethod
    def get_batched_cfg_data(
        self,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None] | None:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...

    @abstractmethod
    def get_cfg_branch_data(
        self, positive: bool
    ) -> tuple[mx.array, mx.array | None, mx.array | None, mx.array | None]:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...


class ModelAdapter(ABC, Generic[ModelT, TransformerT]):
    _config: ImageModelConfig
    _model: ModelT
    _transformer: TransformerT

    @property
    def config(self) -> ImageModelConfig:
        return self._config

    @property
    def model(self) -> ModelT:
        return self._model

    @property
    def transformer(self) -> TransformerT:
        return self._transformer

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def needs_cfg(self) -> bool:
        """此說明已翻譯為繁體中文。"""
        ...

    @abstractmethod
    def _get_latent_creator(self) -> type: ...

    @abstractmethod
    def get_joint_block_wrappers(
        self,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> list["JointBlockWrapper[Any]"]:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...

    @abstractmethod
    def get_single_block_wrappers(
        self,
        text_seq_len: int,
    ) -> list["SingleBlockWrapper[Any]"]:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...

    @abstractmethod
    def slice_transformer_blocks(
        self,
        start_layer: int,
        end_layer: int,
    ):
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...

    def set_image_dimensions(self, image_path: Path) -> tuple[int, int] | None:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        return None

    def create_latents(self, seed: int, runtime_config: Config) -> mx.array:
        """此說明已翻譯為繁體中文。"""
        model: Any = self.model
        return LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
            img2img=Img2Img(
                vae=model.vae,  # 已翻譯註解。
                latent_creator=self._get_latent_creator(),
                sigmas=runtime_config.scheduler.sigmas,  # 已翻譯註解。
                init_time_step=runtime_config.init_time_step,
                image_path=runtime_config.image_path,
            ),
        )

    def decode_latents(
        self,
        latents: mx.array,
        runtime_config: Config,
        seed: int,
        prompt: str,
    ) -> Image.Image:
        model: Any = self.model  # 已翻譯註解。
        latents = self._get_latent_creator().unpack_latents(  # 已翻譯註解。
            latents=latents,
            height=runtime_config.height,
            width=runtime_config.width,
        )
        decoded = model.vae.decode(latents)  # 已翻譯註解。
        # 待辦事項：已翻譯註解。
        # 已翻譯註解。
        # 已翻譯註解。
        generated_image = ImageUtil.to_image(
            decoded_latents=decoded,  # 已翻譯註解。
            config=runtime_config,
            seed=seed,
            prompt=prompt,
            quantization=model.bits,  # 已翻譯註解。
            lora_paths=model.lora_paths,  # 已翻譯註解。
            lora_scales=model.lora_scales,  # 已翻譯註解。
            image_path=runtime_config.image_path,
            image_strength=runtime_config.image_strength,
            generation_time=0,
        )
        return generated_image.image

    @abstractmethod
    def encode_prompt(
        self, prompt: str, negative_prompt: str | None = None
    ) -> "PromptData": ...

    @abstractmethod
    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
    ) -> tuple[mx.array, mx.array]: ...

    @abstractmethod
    def compute_text_embeddings(
        self,
        t: int,
        runtime_config: Config,
        pooled_prompt_embeds: mx.array | None = None,
        hidden_states: mx.array | None = None,
    ) -> mx.array: ...

    @abstractmethod
    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: Config,
        encoder_hidden_states_mask: mx.array | None = None,
        cond_image_grid: tuple[int, int, int]
        | list[tuple[int, int, int]]
        | None = None,
        kontext_image_ids: mx.array | None = None,
    ) -> RotaryEmbeddings: ...

    def merge_streams(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        return mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    @abstractmethod
    def apply_guidance(
        self,
        noise_positive: mx.array,
        noise_negative: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        ...

    def final_projection(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> mx.array:
        transformer: Any = self.transformer
        hidden_states = transformer.norm_out(hidden_states, text_embeddings)  # 已翻譯註解。
        return transformer.proj_out(hidden_states)  # 已翻譯註解。
