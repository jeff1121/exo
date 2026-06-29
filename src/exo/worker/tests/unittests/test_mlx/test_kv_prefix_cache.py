# type: ignore
import time
from typing import cast
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    cache_length,
    encode_prompt,
    get_prefix_length,
    make_kv_cache,
)
from exo.worker.engines.mlx.generator.generate import mlx_generate, prefill
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.tests.unittests.test_mlx.conftest import (
    DEFAULT_GPT_OSS_CONFIG,
    DEFAULT_GPT_OSS_MODEL_ID,
)


def _check_model_exists() -> bool:
    return DEFAULT_GPT_OSS_CONFIG.model_path.exists()


class TestGetPrefixLength:
    def test_identical_arrays(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 5

    def test_no_common_prefix(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        assert get_prefix_length(a, b) == 0

    def test_partial_prefix(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 7, 8])
        assert get_prefix_length(a, b) == 3

    def test_prompt_longer_than_cached(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 3

    def test_cached_longer_than_prompt(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 3

    def test_single_token_match(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 5, 6])
        assert get_prefix_length(a, b) == 1

    def test_empty_prompt(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 0

    def test_empty_cached(self):
        a = mx.array([1, 2, 3])
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0

    def test_both_empty(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0


class TestKVPrefix:
    @pytest.fixture
    def mock_tokenizer(self):
        """此說明已翻譯為繁體中文。"""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        return tokenizer

    def test_starts_empty(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_empties_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.prompts.append(mx.array([1, 2, 3]))
        cache.caches.append([KVCache()])
        cache.clear()
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_on_empty_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.clear()
        assert len(cache.prompts) == 0


def _load_gpt_oss() -> tuple[Model, object]:
    from mlx_lm.utils import load_model

    from exo.worker.engines.mlx.utils_mlx import load_tokenizer_for_model_id

    model_path = DEFAULT_GPT_OSS_CONFIG.model_path
    model_id = ModelId(DEFAULT_GPT_OSS_MODEL_ID)

    model, _ = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    return cast(Model, model), tokenizer


@pytest.mark.slow
@pytest.mark.skipif(
    not _check_model_exists(),
    reason=f"GPT-OSS model not found at {DEFAULT_GPT_OSS_CONFIG.model_path}",
)
class TestKVPrefixCacheWithModel:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        model, tokenizer = _load_gpt_oss()
        return model, tokenizer

    def test_prefill_populates_cache(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello!!")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        # 已翻譯註解。
        assert cache_length(cache) == len(tokens) - 1
        # 已翻譯註解。
        assert len(snapshots) > 0

    def test_add_and_get_exact_match(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Test exact")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        assert len(kv_prefix_cache.prompts) == 1
        stored_length = cache_length(kv_prefix_cache.caches[0])
        assert stored_length > 0

        # 已翻譯註解。
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, tokens
        )
        assert matched_index == 0

        # 已翻譯註解。
        # 已翻譯註解。
        # 已翻譯註解。
        assert len(remaining_tokens) >= 1
        assert mx.array_equal(remaining_tokens, tokens[-len(remaining_tokens) :])

    def test_add_and_get_prefix_match(self, model_and_tokenizer):
        """此說明已翻譯為繁體中文。"""
        model, tokenizer = model_and_tokenizer

        short_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi")],
            max_output_tokens=1,
        )
        short_prompt = apply_chat_template(tokenizer, short_task)
        short_tokens = encode_prompt(tokenizer, short_prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            short_tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(short_tokens, cache, snapshots)

        # 已翻譯註解。
        long_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi there, how are you?")],
            max_output_tokens=1,
        )
        long_prompt = apply_chat_template(tokenizer, long_task)
        long_tokens = encode_prompt(tokenizer, long_prompt)

        # 已翻譯註解。
        expected_prefix = get_prefix_length(long_tokens, short_tokens)
        assert expected_prefix > 0, (
            "Prompts should share a prefix from the chat template"
        )

        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, long_tokens
        )
        assert matched_index == 0

        # 已翻譯註解。
        assert len(remaining_tokens) >= len(long_tokens) - expected_prefix

    def test_stored_cache_not_mutated_after_get_and_generation(
        self, model_and_tokenizer
    ):
        """此說明已翻譯為繁體中文。"""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Mutation test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        # 已翻譯註解。
        result_cache, _, matched_index, _ = kv_prefix_cache.get_kv_cache(model, tokens)
        assert matched_index == 0

        # 已翻譯註解。
        head_dim = result_cache[0].keys.shape[-1]
        num_heads = result_cache[0].keys.shape[1]
        extra_keys = mx.random.normal((1, num_heads, 50, head_dim))
        extra_values = mx.random.normal((1, num_heads, 50, head_dim))
        for layer_cache in result_cache:
            layer_cache.update_and_fetch(extra_keys, extra_values)
        mx.eval([c.keys for c in result_cache])

        # 已翻譯註解。
        assert cache_length(kv_prefix_cache.caches[0]) == stored_length

    def test_stored_cache_survives_repeated_get_mutate_cycles(
        self, model_and_tokenizer
    ):
        """此說明已翻譯為繁體中文。"""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Repeat test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        for i in range(3):
            result_cache, _, _, _ = kv_prefix_cache.get_kv_cache(model, tokens)

            head_dim = result_cache[0].keys.shape[-1]
            num_heads = result_cache[0].keys.shape[1]
            extra = mx.random.normal((1, num_heads, 30, head_dim))
            for layer_cache in result_cache:
                layer_cache.update_and_fetch(extra, extra)
            mx.eval([c.keys for c in result_cache])

            assert cache_length(kv_prefix_cache.caches[0]) == stored_length, (
                f"Failed on loop {i}"
            )

    def test_mlx_generate_populates_cache(self, model_and_tokenizer):
        """此說明已翻譯為繁體中文。"""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # 已翻譯註解。
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1
        assert len(kv_prefix_cache.caches) == 1
        # 已翻譯註解。
        # 已翻譯註解。
        # 已翻譯註解。
        assert cache_length(kv_prefix_cache.caches[0]) == len(prompt_tokens) - 2

    def test_mlx_generate_second_call_gets_prefix_hit(self, model_and_tokenizer):
        """此說明已翻譯為繁體中文。"""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Reuse test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # 已翻譯註解。
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1

        # 已翻譯註解。
        # 已翻譯註解。
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        # 已翻譯註解。
        # 已翻譯註解。
        assert matched_index == 0
        # 已翻譯註解。
        assert len(remaining_tokens) == 2
        assert mx.array_equal(remaining_tokens, prompt_tokens[-2:])

    def test_mlx_generate_long_prompt_updates_cache_in_place(self, model_and_tokenizer):
        """此說明已翻譯為繁體中文。"""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # 已翻譯註解。
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        repeats = (1200 // len(base_tokens)) + 2
        long_content = base_text * repeats

        task1 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content=long_content)],
            max_output_tokens=5,
        )
        prompt1 = apply_chat_template(tokenizer, task1)
        prompt1_tokens = encode_prompt(tokenizer, prompt1)
        assert len(prompt1_tokens) > 1000, (
            "Prompt must exceed _MIN_PREFIX_HIT_TO_UPDATE"
        )

        # 已翻譯註解。
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task1,
            prompt=prompt1,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass
        first_gen_time = time.perf_counter() - t0

        assert len(kv_prefix_cache.prompts) == 1
        first_cache_length = cache_length(kv_prefix_cache.caches[0])

        # 已翻譯註解。
        task2 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[
                InputMessage(role="user", content=long_content),
                InputMessage(role="assistant", content="Sure, I can help."),
                InputMessage(role="user", content="Tell me more."),
            ],
            max_output_tokens=5,
        )
        prompt2 = apply_chat_template(tokenizer, task2)
        prompt2_tokens = encode_prompt(tokenizer, prompt2)

        # 已翻譯註解。
        prefix_len = get_prefix_length(prompt2_tokens, prompt1_tokens)
        assert prefix_len > 1000, "Prompts must share > 1000 token prefix"

        # 已翻譯註解。
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task2,
            prompt=prompt2,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass
        second_gen_time = time.perf_counter() - t0

        # 已翻譯註解。
        assert second_gen_time < first_gen_time * 0.5, (
            f"Expected prefix cache speedup: "
            f"first={first_gen_time:.2f}s, second={second_gen_time:.2f}s"
        )

        # 已翻譯註解。
        assert len(kv_prefix_cache.prompts) == 1
        # 已翻譯註解。
        updated_cache_length = cache_length(kv_prefix_cache.caches[0])
        assert updated_cache_length > first_cache_length

    def test_mlx_generate_stored_cache_not_mutated(self, model_and_tokenizer):
        """此說明已翻譯為繁體中文。"""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Immutable test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)

        # 已翻譯註解。
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        firstcache_length = cache_length(kv_prefix_cache.caches[0])

        # 已翻譯註解。
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        # 已翻譯註解。
        assert cache_length(kv_prefix_cache.caches[0]) == firstcache_length

    def test_evicts_lru_entry_under_memory_pressure(self, model_and_tokenizer):
        """此說明已翻譯為繁體中文。"""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # 已翻譯註解。
        prompts = ["First entry", "Second entry", "Third entry"]
        for i, content in enumerate(prompts):
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content=content)],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)
            # 已翻譯註解。
            kv_prefix_cache._last_used[i] = float(i)

        assert len(kv_prefix_cache.prompts) == 3

        # 已翻譯註解。
        kv_prefix_cache._last_used[2] = 100.0
        # 已翻譯註解。

        # 已翻譯註解。
        with patch(
            "exo.worker.engines.mlx.cache.get_memory_used_percentage",
            return_value=0.95,
        ):
            # 已翻譯註解。
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content="New entry")],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)

        # 已翻譯註解。
        # 已翻譯註解。
        # 已翻譯註解。
        assert len(kv_prefix_cache.prompts) == 1
        # 已翻譯註解。
        assert get_prefix_length(kv_prefix_cache.prompts[0], tokens) == len(tokens)
