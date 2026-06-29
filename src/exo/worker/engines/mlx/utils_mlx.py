import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import VisionProcessor

# 已翻譯註解。
# 已翻譯註解。
# 已翻譯註解。
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # 已翻譯註解。

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelId
from exo.worker.engines.mlx.constants import TRUST_REMOTE_CODE

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.download.download_utils import build_model_path
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import ChatTemplateValue, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.auto_parallel import (
    get_inner_model,
    get_layers,
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.bootstrap import logger


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_card.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    """
    此說明已翻譯為繁體中文。
    """
    rank = bound_instance.bound_shard.device_rank
    logger.info(f"Starting initialization for rank {rank}")

    with tempfile.TemporaryDirectory() as tmpdir:
        coordination_file = str(
            Path(tmpdir) / f"hosts_{bound_instance.instance.instance_id}_{rank}.json"
        )
        # 待辦事項：已翻譯註解。
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                # 已翻譯註解。

                group = mx.distributed.init(backend="ring", strict=True)

            case MlxJacclInstance(
                jaccl_devices=jaccl_devices, jaccl_coordinators=jaccl_coordinators
            ):
                assert all(
                    jaccl_devices[i][i] is None for i in range(len(jaccl_devices))
                )
                # 已翻譯註解。
                jaccl_devices_json = json.dumps(jaccl_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(jaccl_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(
                    f"rank {rank} MLX_IBV_DEVICES: {coordination_file} with devices: {jaccl_devices_json}"
                )
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator
                group = mx.distributed.init(backend="jaccl", strict=True)

        logger.info(f"Rank {rank} mlx distributed initialization complete")

        return group


def initialize_mlx(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    # 已翻譯註解。
    # 待辦事項：已翻譯註解。
    mx.random.seed(42)

    assert len(bound_instance.instance.shard_assignments.node_to_runner) > 1, (
        "Tried to initialize mlx for a single node instance"
    )
    return mlx_distributed_init(bound_instance)


def load_mlx_items(
    bound_instance: BoundInstance,
    group: mx.distributed.Group | None,
) -> Generator[
    ModelLoadingResponse, None, tuple[Model, TokenizerWrapper, "VisionProcessor | None"]
]:
    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(bound_instance.bound_shard.model_card.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, lazy=True, strict=False)
        # 已翻譯註解。
        try:
            inner = get_inner_model(model)
            layers = get_layers(inner)
            total = len(layers)
            for i, layer in enumerate(layers):
                mx.eval(layer)  # type: ignore
                yield ModelLoadingResponse(layers_loaded=i, total=total)
        except ValueError as e:
            logger.opt(exception=e).debug(
                "Model architecture doesn't support layer-by-layer progress tracking",
            )
        mx.eval(model)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = yield from shard_and_load(
            bound_instance.bound_shard,
            group=group,
        )
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

    mx.clear_cache()

    vision_config = bound_instance.bound_shard.model_card.vision

    if vision_config is not None:
        from exo.worker.engines.mlx.vision import VisionProcessor

        vision_start_time = time.perf_counter()
        try:
            vision_processor: VisionProcessor | None = VisionProcessor(
                vision_config, bound_instance.bound_shard.model_card.model_id
            )
            vision_processor.load()
            logger.info(
                f"Time taken to load vision weights: {(time.perf_counter() - vision_start_time):.2f}s"
            )
        except Exception as e:
            logger.opt(exception=e).error(
                "Failed to load vision weights — disabling vision for this runner"
            )
            vision_processor = None
    else:
        vision_processor = None

    return cast(Model, model), tokenizer, vision_processor


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: mx.distributed.Group,
) -> Generator[ModelLoadingResponse, None, tuple[nn.Module, TokenizerWrapper]]:
    model_path = build_model_path(shard_metadata.model_card.model_id)

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # 待辦事項：已翻譯註解。
        # 已翻譯註解。
        #     已翻譯註解。

        #     已翻譯註解。

        # 已翻譯註解。
        #     已翻譯註解。
        #         已翻譯註解。

        #     已翻譯註解。
        # 已翻譯註解。
        #        已翻譯註解。
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = yield from tensor_auto_parallel(model, group)
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = yield from pipeline_auto_parallel(model, group, shard_metadata)
            mx.eval(model.parameters())
        case CfgShardMetadata():
            raise ValueError(
                "CfgShardMetadata is not supported for text model loading - "
                "this metadata type is only for image generation models"
            )

    # 待辦事項：已翻譯註解。
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # 已翻譯註解。
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """此說明已翻譯為繁體中文。"""
    return load_tokenizer_for_model_id(
        shard_metadata.model_card.model_id,
        model_path,
        trust_remote_code=shard_metadata.model_card.trust_remote_code,
    )


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-5" in model_id_lower:
        # 已翻譯註解。
        return [154820, 154827, 154829]
    elif "glm" in model_id_lower:
        # 已翻譯註解。
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower:
        return [200002, 200012]
    elif (
        "qwen3.5" in model_id_lower
        or "qwen-3.5" in model_id_lower
        or "qwen3.6" in model_id_lower
        or "qwen-3.6" in model_id_lower
    ):
        # 已翻譯註解。
        return [248046, 248044]
    elif "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    return None


def load_tokenizer_for_model_id(
    model_id: ModelId, model_path: Path, *, trust_remote_code: bool = TRUST_REMOTE_CODE
) -> TokenizerWrapper:
    """
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。
    """
    model_id_lower = model_id.lower()
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # 已翻譯註解。
    if "kimi-k2" in model_id_lower:
        import importlib.util
        import types

        sys.path.insert(0, str(model_path))

        # 已翻譯註解。
        tool_decl_path = model_path / "tool_declaration_ts.py"
        if tool_decl_path.exists():
            spec = importlib.util.spec_from_file_location(
                "tool_declaration_ts", tool_decl_path
            )
            if spec and spec.loader:
                tool_decl_module = importlib.util.module_from_spec(spec)
                sys.modules["tool_declaration_ts"] = tool_decl_module
                spec.loader.exec_module(tool_decl_module)

        # 已翻譯註解。
        tok_path = model_path / "tokenization_kimi.py"
        source = tok_path.read_text()
        source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
        spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
        if spec:
            tok_module = types.ModuleType("tokenization_kimi")
            tok_module.__file__ = str(tok_path)
            sys.modules["tokenization_kimi"] = tok_module
            exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
            TikTokenTokenizer = tok_module.TikTokenTokenizer  # type: ignore[attr-defined]  # noqa: N806
        else:
            from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # 已翻譯註解。

        # 已翻譯註解。
        # 已翻譯註解。
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # 已翻譯註解。
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # 已翻譯註解。

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(
            hf_tokenizer,
            eos_token_ids=eos_token_ids,
            tool_call_start="<|tool_calls_section_begin|>",
            tool_call_end="<|tool_calls_section_end|>",
            tool_parser=_parse_kimi_tool_calls,
        )

    # 已翻譯註解。
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": trust_remote_code},
        eos_token_ids=eos_token_ids,
    )

    return tokenizer


def _normalize_tool_calls(msg_dict: dict[str, Any]) -> None:
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    tool_calls = msg_dict.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return

    for tc in tool_calls:  # 已翻譯註解。
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")  # 已翻譯註解。
        if not isinstance(func, dict):
            continue
        args = func.get("arguments")  # 已翻譯註解。
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                func["arguments"] = json.loads(args)


def _collect_nested_property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    properties: dict[str, Any] = schema.get("properties", {})  # type: ignore[reportAny]
    for prop_spec in properties.values():  # 已翻譯註解。
        if not isinstance(prop_spec, dict):
            continue
        if prop_spec.get("type") == "array":  # type: ignore[reportAny]
            items: dict[str, Any] | None = prop_spec.get("items")  # type: ignore[reportAny]
            if isinstance(items, dict) and items.get("type") == "object":  # type: ignore[reportAny]
                inner_props: dict[str, Any] = items.get("properties", {})  # type: ignore[reportAny]
                for k in inner_props:  # 已翻譯註解。
                    names.add(str(k))  # 已翻譯註解。
                names.update(_collect_nested_property_names(items))  # 已翻譯註解。
    return names


def _schemas_lost_in_prompt(prompt: str, tools: list[dict[str, Any]]) -> bool:
    """此說明已翻譯為繁體中文。"""
    for tool in tools:
        fn: dict[str, Any] = tool.get("function", {})  # type: ignore
        params: dict[str, Any] = fn.get("parameters", {})  # type: ignore
        nested = _collect_nested_property_names(params)
        if nested and not all(name in prompt for name in nested):
            return True
    return False


_LOSSY_TEMPLATE_PATTERN = re.compile(
    r"""inner_type\s*==\s*["']object \| object["']\s*or\s*inner_type\|length\s*>\s*\d+""",
)


def _patch_lossy_chat_template(template: str) -> str | None:
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    patched, n = _LOSSY_TEMPLATE_PATTERN.subn(
        lambda m: m.group(0).split(" or ")[0],  # 已翻譯註解。
        template,
    )
    return patched if n > 0 else None


def _needs_dsml_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v3.2" in task_params.model.lower()


def _needs_v4_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v4" in task_params.model.lower()


def _v4_reasoning_effort(task_params: TextGenerationTaskParams) -> str | None:
    effort = task_params.reasoning_effort
    if effort == "xhigh":
        return "max"
    if effort == "high":
        return "high"
    return None


def _strip_v4_thinking_markers(content: str) -> str:
    """此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。"""
    block = re.compile(r"<think>.*?</think>", re.DOTALL)
    if not content:
        return content
    cleaned = block.sub("", content)
    return cleaned.replace("<think>", "").replace("</think>", "")


def consolidate_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    """
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            content = cast(str, msg.get("content", ""))
            if content:
                system_parts.append(content)
        else:
            non_system.append(msg)
    formatted_messages = non_system
    if system_parts:
        formatted_messages.insert(
            0, {"role": "system", "content": "\n".join(system_parts)}
        )
    return formatted_messages


def render_chat_template(
    tokenizer: TokenizerWrapper,
    messages: list[dict[str, Any]],
    task_params: TextGenerationTaskParams,
) -> str:
    """
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    formatted_messages = consolidate_system_messages(messages)

    # 已翻譯註解。
    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        partial_assistant_content = cast(str, formatted_messages[-1].get("content", ""))
        formatted_messages = formatted_messages[:-1]

    if _needs_dsml_encoding(task_params):
        from exo.worker.engines.mlx.vendor.dsml_encoding import encode_messages

        prompt = encode_messages(
            messages=formatted_messages,
            # 已翻譯註解。
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            tools=task_params.tools,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    if _needs_v4_encoding(task_params):
        from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
            encode_messages as encode_messages_v4,
        )

        v4_messages = [dict(m) for m in formatted_messages]
        for msg in v4_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = _strip_v4_thinking_markers(content)
        if task_params.tools:
            for msg in v4_messages:
                if msg.get("role") in ("system", "developer"):
                    msg["tools"] = task_params.tools
                    break
            else:
                v4_messages.insert(
                    0, {"role": "system", "content": "", "tools": task_params.tools}
                )

        prompt = encode_messages_v4(
            messages=v4_messages,
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            reasoning_effort=_v4_reasoning_effort(task_params),
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    for msg in formatted_messages:
        _normalize_tool_calls(msg)

    # 已翻譯註解。
    if "gpt-oss" in task_params.model.lower():
        for msg in formatted_messages:
            if msg.get("role") == "assistant" and "thinking" not in msg:
                rc = msg.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    msg["thinking"] = rc

    extra_kwargs: dict[str, Any] = {}
    if task_params.enable_thinking is not None:
        # 已翻譯註解。
        # 已翻譯註解。
        extra_kwargs["enable_thinking"] = task_params.enable_thinking
        extra_kwargs["thinking"] = task_params.enable_thinking
    if task_params.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = task_params.reasoning_effort

    patched_template: str | None = None
    if task_params.tools:
        original_template: str | None = getattr(tokenizer, "chat_template", None)
        if isinstance(original_template, str):
            patched_template = _patch_lossy_chat_template(original_template)
            if patched_template is not None:
                logger.info(
                    "Patched lossy chat template (removed inner_type length guard)"
                )

    prompt: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=task_params.tools,
        **({"chat_template": patched_template} if patched_template is not None else {}),
        **extra_kwargs,
    )

    if task_params.tools and _schemas_lost_in_prompt(prompt, task_params.tools):
        logger.warning("Chat template lost nested tool schemas even after patching")

    if partial_assistant_content:
        prompt += partial_assistant_content

    return prompt


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    task_params: TextGenerationTaskParams,
) -> str:
    messages: list[dict[str, ChatTemplateValue]] = []
    if task_params.chat_template_messages is not None:
        # 已翻譯註解。
        messages = task_params.chat_template_messages
    else:
        # 已翻譯註解。
        if task_params.instructions:
            messages.append({"role": "system", "content": task_params.instructions})

        # 已翻譯註解。
        for msg in task_params.input:
            if not msg.content:
                logger.warning("Received message with empty content, skipping")
                continue
            messages.append({"role": msg.role, "content": msg.content})

    prompt = render_chat_template(tokenizer, messages, task_params)
    logger.debug(prompt)

    return prompt


def system_prompt_token_count(
    task_params: TextGenerationTaskParams,
    tokenizer: TokenizerWrapper,
) -> int:
    """此說明已翻譯為繁體中文。"""
    parts: list[str] = []
    if task_params.chat_template_messages is not None:
        for msg in task_params.chat_template_messages:
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
    else:
        if task_params.instructions:
            parts.append(task_params.instructions)
        for msg in task_params.input:
            if msg.role in ("system", "developer"):
                parts.append(msg.content)
    if len(parts) == 0:
        return 0
    return len(tokenizer.encode(" ".join(parts), add_special_tokens=False))


def detect_thinking_prompt_suffix(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    think_token = tokenizer.think_start

    return think_token is not None and prompt.rstrip().endswith(think_token)


def fix_unmatched_think_end_tokens(
    tokens: mx.array, tokenizer: TokenizerWrapper
) -> mx.array:
    if not tokenizer.has_thinking:
        return tokens
    assert tokenizer.think_start_tokens
    assert tokenizer.think_end_tokens
    think_start_tokens: list[int] = tokenizer.think_start_tokens
    think_end_tokens: list[int] = tokenizer.think_end_tokens
    token_list: list[int] = cast(list[int], tokens.tolist())
    result: list[int] = []

    depth = 0
    accumulated_think_start_length = 0
    accumulated_think_end_length = 0

    for token in token_list:
        if token == think_start_tokens[accumulated_think_start_length]:
            accumulated_think_start_length += 1
            if accumulated_think_start_length == len(think_start_tokens):
                depth += 1
                accumulated_think_start_length = 0

        elif token == think_end_tokens[accumulated_think_end_length]:
            accumulated_think_end_length += 1
            if accumulated_think_end_length == len(think_end_tokens):
                if depth == 0:
                    result.extend(think_start_tokens)
                else:
                    depth -= 1
                accumulated_think_end_length = 0

        else:
            accumulated_think_start_length = 0
            accumulated_think_end_length = 0

        result.append(token)
    return mx.array(result)


class NullKVCache(KVCache):
    """
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # 已翻譯註解。
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # 已翻譯註解。
        assert self.keys is not None and self.values is not None
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def mlx_force_oom(size: int = 200000) -> None:
    """
    此說明已翻譯為繁體中文。
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    if not mx.metal.is_available():
        return

    max_rec_size = Memory.from_bytes(
        int(mx.device_info()["max_recommended_working_set_size"])
    )
    if model_size > 0.9 * max_rec_size:
        logger.warning(
            f"Generating with a model that requires {model_size.in_float_mb:.1f} MB "
            f"which is close to the maximum recommended size of {max_rec_size.in_float_mb:.1f} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    mx.set_wired_limit(max_rec_size.in_bytes)
    logger.info(f"Wired limit set to {max_rec_size}.")


def mlx_cleanup(
    model: Model | None,
    tokenizer: TokenizerWrapper | None,
    group: mx.distributed.Group | None,
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()


def mx_any(bool_: bool, group: mx.distributed.Group | None) -> bool:
    if group is None:
        return bool_
    num_true = mx.distributed.all_sum(
        mx.array(bool_), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(num_true)
    return num_true.item() > 0


def mx_barrier(group: mx.distributed.Group | None):
    if group is None:
        return
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
        )
    )


def _parse_kimi_tool_calls(text: str):
    import regex as re

    # 已翻譯註解。
    #   已翻譯註解。
    _func_name_regex = re.compile(
        r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
    _tool_call_split_regex = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
    )

    def _parse_single_tool(text: str) -> dict[str, Any]:
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError("No tool call found.")
        tool_call_id = func_name_match.group(1)  # 已翻譯註解。
        func_name = func_name_match.group(2)  # 已翻譯註解。

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError("No tool call arguments found.")
        func_args = func_args_match.group(1)
        arg_dct = json.loads(func_args)  # 已翻譯註解。

        return dict(id=tool_call_id, name=func_name, arguments=arg_dct)  # 已翻譯註解。

    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]  # 已翻譯註解。
    else:
        return [_parse_single_tool(text)]


def mx_all_gather_tasks(
    tasks: list[TextGeneration],
    group: mx.distributed.Group | None,
) -> tuple[list[TextGeneration], list[TextGeneration]]:
    def encode_task_id(task_id: TaskId) -> list[int]:
        utf8_task_id = task_id.encode()
        return [
            int.from_bytes(utf8_task_id[i : i + 1]) for i in range(len(utf8_task_id))
        ]

    def decode_task_id(encoded_task_id: list[int]) -> TaskId:
        return TaskId(
            bytes.decode(b"".join((x).to_bytes(length=1) for x in encoded_task_id))
        )

    uuid_byte_length = 36

    n_tasks = len(tasks)
    all_counts = cast(
        list[int],
        mx.distributed.all_gather(mx.array([n_tasks]), group=group).tolist(),
    )
    max_tasks = max(all_counts)
    world_size: int = 1 if group is None else group.size()

    if max_tasks == 0:
        return [], []

    padded = [encode_task_id(task.task_id) for task in tasks] + [
        [0] * uuid_byte_length
    ] * (max_tasks - n_tasks)

    assert all(len(encoded_task_id) == uuid_byte_length for encoded_task_id in padded)

    gathered = cast(
        list[list[list[int]]],
        mx.distributed.all_gather(mx.array(padded), group=group)
        .reshape(world_size, max_tasks, -1)
        .tolist(),
    )
    all_task_ids: list[list[TaskId]] = [
        [decode_task_id(encoded_task_id) for encoded_task_id in rank_tasks[:count]]
        for rank_tasks, count in zip(gathered, all_counts, strict=True)
    ]

    agreed_ids = set[TaskId].intersection(*(set(tids) for tids in all_task_ids))

    local_tasks = {task.task_id: task for task in tasks}
    agreed = [local_tasks[tid] for tid in sorted(agreed_ids)]
    different = [task for task in tasks if task.task_id not in agreed_ids]
    return agreed, different
