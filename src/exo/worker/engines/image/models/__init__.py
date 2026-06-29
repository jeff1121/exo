from pathlib import Path
from typing import Any, Callable

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import ModelAdapter
from exo.worker.engines.image.models.flux import (
    FLUX_DEV_CONFIG,
    FLUX_KONTEXT_CONFIG,
    FLUX_SCHNELL_CONFIG,
    FluxKontextModelAdapter,
    FluxModelAdapter,
)
from exo.worker.engines.image.models.qwen import (
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_CONFIG,
    QwenEditModelAdapter,
    QwenModelAdapter,
)

__all__: list[str] = []

# 已翻譯註解。
# 已翻譯註解。
AdapterFactory = Callable[
    [ImageModelConfig, str, Path, int | None], ModelAdapter[Any, Any]
]

# 已翻譯註解。
_ADAPTER_REGISTRY: dict[str, AdapterFactory] = {
    "flux": FluxModelAdapter,
    "flux-kontext": FluxKontextModelAdapter,
    "qwen-edit": QwenEditModelAdapter,
    "qwen": QwenModelAdapter,
}

# 已翻譯註解。
# 已翻譯註解。
_CONFIG_REGISTRY: dict[str, ImageModelConfig] = {
    "flux.1-schnell": FLUX_SCHNELL_CONFIG,
    "flux.1-kontext": FLUX_KONTEXT_CONFIG,  # 已翻譯註解。
    "flux.1-krea-dev": FLUX_DEV_CONFIG,  # 已翻譯註解。
    "flux.1-dev": FLUX_DEV_CONFIG,
    "qwen-image-edit": QWEN_IMAGE_EDIT_CONFIG,  # 已翻譯註解。
    "qwen-image": QWEN_IMAGE_CONFIG,
}


def get_config_for_model(model_id: str) -> ImageModelConfig:
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。
    """
    model_id_lower = model_id.lower()

    for pattern, config in _CONFIG_REGISTRY.items():
        if pattern in model_id_lower:
            return config

    raise ValueError(f"No configuration found for model: {model_id}")


def create_adapter_for_model(
    config: ImageModelConfig,
    model_id: str,
    local_path: Path,
    quantize: int | None = None,
) -> ModelAdapter[Any, Any]:
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
    """
    factory = _ADAPTER_REGISTRY.get(config.model_family)
    if factory is None:
        raise ValueError(f"No adapter found for model family: {config.model_family}")
    return factory(config, model_id, local_path, quantize)
