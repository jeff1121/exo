import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Generator, Iterable

import aiofiles
import aiofiles.os as aios
from loguru import logger

from exo.shared.types.worker.shards import ShardMetadata


def filter_repo_objects[T](
    items: Iterable[T],
    *,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    key: Callable[[T], str] | None = None,
) -> Generator[T, None, None]:
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]
    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]
    if allow_patterns is not None:
        allow_patterns = [_add_wildcard_to_directories(p) for p in allow_patterns]
    if ignore_patterns is not None:
        ignore_patterns = [_add_wildcard_to_directories(p) for p in ignore_patterns]

    if key is None:

        def _identity(item: T) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, Path):
                return str(item)
            raise ValueError(
                f"Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string."
            )

        key = _identity

    for item in items:
        path = key(item)
        if allow_patterns is not None and not any(
            fnmatch(path, r) for r in allow_patterns
        ):
            continue
        if ignore_patterns is not None and any(
            fnmatch(path, r) for r in ignore_patterns
        ):
            continue
        yield item


def _add_wildcard_to_directories(pattern: str) -> str:
    if pattern[-1] == "/":
        return pattern + "*"
    return pattern


def get_hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def get_hf_home() -> Path:
    """此說明已翻譯為繁體中文。"""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


async def get_hf_token() -> str | None:
    """此說明已翻譯為繁體中文。"""
    # 先檢查環境變數
    if token := os.environ.get("HF_TOKEN"):
        return token
    # 若無則回退為檔案型權杖
    token_path = get_hf_home() / "token"
    if await aios.path.exists(token_path):
        async with aiofiles.open(token_path, "r") as f:
            return (await f.read()).strip()
    return None


async def get_auth_headers() -> dict[str, str]:
    """若有權杖可用則取得驗證標頭。"""
    token = await get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def extract_layer_num(tensor_name: str) -> int | None:
    # 這是簡化範例，可能需依實際命名慣例調整
    parts = tensor_name.split(".")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def get_allow_patterns(weight_map: dict[str, str], shard: ShardMetadata) -> list[str]:
    default_patterns = set(
        [
            "*.json",
            "*.py",
            "tokenizer.model",
            "tiktoken.model",
            "*/spiece.model",
            "*.tiktoken",
            "*.txt",
            "*.jinja",
        ]
    )
    shard_specific_patterns: set[str] = set()

    if shard.model_card.components is not None:
        shardable_component = next(
            (c for c in shard.model_card.components if c.can_shard), None
        )

        if weight_map and shardable_component:
            for tensor_name, filename in weight_map.items():
                # 從張量名稱移除元件前綴（由權重映射命名空間加入）
                # 已翻譯註解。
                if "/" in tensor_name:
                    _, tensor_name_no_prefix = tensor_name.split("/", 1)
                else:
                    tensor_name_no_prefix = tensor_name

                # 由檔名判斷此檔案所屬元件
                component_path = Path(filename).parts[0] if "/" in filename else None

                if component_path == shardable_component.component_path.rstrip("/"):
                    layer_num = extract_layer_num(tensor_name_no_prefix)
                    if (
                        layer_num is not None
                        and shard.start_layer <= layer_num < shard.end_layer
                    ):
                        shard_specific_patterns.add(filename)

                    if shard.is_first_layer or shard.is_last_layer:
                        shard_specific_patterns.add(filename)
                else:
                    shard_specific_patterns.add(filename)

        else:
            shard_specific_patterns = set(["*.safetensors"])

        # 待辦事項：已翻譯註解。
        for component in shard.model_card.components:
            if not component.can_shard and component.safetensors_index_filename is None:
                component_pattern = f"{component.component_path.rstrip('/')}/*"
                shard_specific_patterns.add(component_pattern)
    else:
        if weight_map:
            for tensor_name, filename in weight_map.items():
                layer_num = extract_layer_num(tensor_name)
                if (
                    layer_num is not None
                    and shard.start_layer <= layer_num < shard.end_layer
                ):
                    shard_specific_patterns.add(filename)
            layer_independent_files = set(
                [v for k, v in weight_map.items() if extract_layer_num(k) is None]
            )
            shard_specific_patterns.update(layer_independent_files)
            logger.debug(f"get_allow_patterns {shard=} {layer_independent_files=}")
        else:
            shard_specific_patterns = set(["*.safetensors"])

    logger.info(f"get_allow_patterns {shard=} {shard_specific_patterns=}")
    return list(default_patterns | shard_specific_patterns)
