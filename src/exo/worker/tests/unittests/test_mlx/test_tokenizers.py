"""
此說明已翻譯為繁體中文。

此說明已翻譯為繁體中文。
此說明已翻譯為繁體中文。
"""

import asyncio
import contextlib
from pathlib import Path

import pytest

from exo.download.download_utils import (
    download_file_with_retry,
    fetch_file_list_with_cache,
    resolve_model_dir,
)
from exo.shared.models.model_cards import ModelCard, ModelId, card_cache
from exo.worker.engines.mlx.utils_mlx import (
    get_eos_token_ids_for_model,
    load_tokenizer_for_model_id,
)

# 已翻譯註解。
TOKENIZER_FILE_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tiktoken.model",
    "added_tokens.json",
    "tokenizer.model",
    "tokenization_*.py",  # 已翻譯註解。
    "tool_declaration_ts.py",  # 已翻譯註解。
]


def is_tokenizer_file(filename: str) -> bool:
    """此說明已翻譯為繁體中文。"""
    for pattern in TOKENIZER_FILE_PATTERNS:
        if "*" in pattern:
            prefix = pattern.split("*")[0]
            suffix = pattern.split("*")[1]
            if filename.startswith(prefix) and filename.endswith(suffix):
                return True
        elif filename == pattern:
            return True
    return False


async def download_tokenizer_files(model_id: ModelId) -> Path:
    """此說明已翻譯為繁體中文。"""
    target_dir = await resolve_model_dir(model_id)

    file_list = await fetch_file_list_with_cache(model_id, "main", recursive=True)

    tokenizer_files = [f for f in file_list if is_tokenizer_file(f.path)]

    if not tokenizer_files:
        pytest.skip(f"No tokenizer files found for {model_id}")

    for file_entry in tokenizer_files:
        with contextlib.suppress(FileNotFoundError):
            await download_file_with_retry(
                model_id, "main", file_entry.path, target_dir
            )

    return target_dir


# 已翻譯註解。
def get_test_models() -> list[ModelCard]:
    """此說明已翻譯為繁體中文。"""
    # 已翻譯註解。
    families: dict[str, ModelCard] = {}
    for card in asyncio.run(card_cache.list_all()):
        # 已翻譯註解。
        parts = card.model_id.short().split("-")
        family = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]

        if family not in families:
            families[family] = card

    return list(families.values())


TEST_MODELS: list[ModelCard] = get_test_models()

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def event_loop():
    """此說明已翻譯為繁體中文。"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.parametrize(
    "model_card",
    TEST_MODELS,
)
@pytest.mark.asyncio
async def test_tokenizer_encode_decode(model_card: ModelCard) -> None:
    """此說明已翻譯為繁體中文。"""
    model_id = model_card.model_id

    # 已翻譯註解。
    model_path = await download_tokenizer_files(model_id)

    # 已翻譯註解。
    has_tokenizer = (
        (model_path / "tokenizer.json").exists()
        or (model_path / "tokenizer_config.json").exists()
        or (model_path / "tiktoken.model").exists()
        or (model_path / "tokenizer.model").exists()
    )
    if not has_tokenizer:
        pytest.skip(f"Required tokenizer files not found for {model_id}")

    # 已翻譯註解。
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)

    # 已翻譯註解。
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    assert isinstance(encoded, list), f"encode() should return a list for {model_id}"
    assert len(encoded) > 0, f"encode() should return non-empty list for {model_id}"
    assert all(isinstance(t, int) for t in encoded), (
        f"All tokens should be integers for {model_id}"
    )

    # 已翻譯註解。
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str), f"decode() should return a string for {model_id}"
    normalized_decoded = decoded.replace(" ", "").lower()
    normalized_expected = test_text.replace(" ", "").lower()
    assert normalized_expected in normalized_decoded, (
        f"decode(encode(x)) should preserve text for {model_id}: got {decoded!r}"
    )

    # 已翻譯註解。
    long_text = "The quick brown fox jumps over the lazy dog. " * 10
    long_encoded = tokenizer.encode(long_text)
    assert len(long_encoded) > len(encoded), (
        f"Longer text should produce more tokens for {model_id}"
    )

    # 已翻譯註解。
    empty_encoded = tokenizer.encode("")
    assert isinstance(empty_encoded, list), (
        f"encode('') should return a list for {model_id}"
    )

    # 已翻譯註解。
    special_text = 'Hello!\n\tWorld? <test> & "quotes"'
    special_encoded = tokenizer.encode(special_text)
    assert len(special_encoded) > 0, f"Special chars should encode for {model_id}"

    # 已翻譯註解。
    unicode_text = "Hello 世界 🌍"
    unicode_encoded = tokenizer.encode(unicode_text)
    assert len(unicode_encoded) > 0, f"Unicode should encode for {model_id}"


@pytest.mark.parametrize(
    "model_card",
    TEST_MODELS,
)
@pytest.mark.asyncio
async def test_tokenizer_has_required_attributes(model_card: ModelCard) -> None:
    """此說明已翻譯為繁體中文。"""
    model_id = model_card.model_id

    model_path = await download_tokenizer_files(model_id)

    has_tokenizer = (
        (model_path / "tokenizer.json").exists()
        or (model_path / "tokenizer_config.json").exists()
        or (model_path / "tiktoken.model").exists()
        or (model_path / "tokenizer.model").exists()
    )
    if not has_tokenizer:
        pytest.skip(f"Required tokenizer files not found for {model_id}")

    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # 已翻譯註解。
    empty_vocab: dict[str, int] = {}
    vocab_size: int = getattr(tokenizer, "vocab_size", None) or len(
        getattr(tokenizer, "get_vocab", lambda: empty_vocab)()
    )
    assert vocab_size > 0, f"Tokenizer should have vocab_size > 0 for {model_id}"

    # 已翻譯註解。
    has_eos = (
        eos_token_ids is not None
        or getattr(tokenizer, "eos_token_id", None) is not None
        or getattr(tokenizer, "eos_token", None) is not None
    )
    assert has_eos, f"Tokenizer should have EOS token for {model_id}"


@pytest.mark.parametrize(
    "model_card",
    TEST_MODELS,
)
@pytest.mark.asyncio
async def test_tokenizer_special_tokens(model_card: ModelCard) -> None:
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    model_id = model_card.model_id

    model_path = await download_tokenizer_files(model_id)

    has_tokenizer = (
        (model_path / "tokenizer.json").exists()
        or (model_path / "tokenizer_config.json").exists()
        or (model_path / "tiktoken.model").exists()
        or (model_path / "tokenizer.model").exists()
    )
    assert has_tokenizer, f"Required tokenizer files not found for {model_id}"

    tokenizer = load_tokenizer_for_model_id(model_id, model_path)

    # 已翻譯註解。
    special_tokens: list[str] = []

    # 已翻譯註解。
    if hasattr(tokenizer, "all_special_tokens"):
        special_tokens.extend(tokenizer.all_special_tokens)
    elif hasattr(tokenizer, "_tokenizer") and hasattr(
        tokenizer._tokenizer,
        "all_special_tokens",
    ):
        special_tokens.extend(tokenizer._tokenizer.all_special_tokens)

    # 已翻譯註解。
    for attr in [
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "sep_token",
        "cls_token",
    ]:
        token = getattr(tokenizer, attr, None)
        if token is None and hasattr(tokenizer, "_tokenizer"):
            token = getattr(tokenizer._tokenizer, attr, None)
        if token and isinstance(token, str) and token not in special_tokens:
            special_tokens.append(token)

    # 已翻譯註解。
    if special_tokens:
        # 已翻譯註解。
        test_with_special = f"{special_tokens[0]}Hello world"
        if len(special_tokens) > 1:
            test_with_special += f"{special_tokens[1]}"

        encoded = tokenizer.encode(test_with_special)
        assert isinstance(encoded, list), (
            f"encode() with special tokens should return list for {model_id}"
        )
        assert len(encoded) > 0, (
            f"encode() with special tokens should return non-empty list for {model_id}"
        )
        assert all(isinstance(t, int) for t in encoded), (
            f"All tokens should be integers for {model_id}"
        )

        # 已翻譯註解。
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str), f"decode() should return string for {model_id}"

    # 已翻譯註解。
    # 已翻譯註解。
    angle_bracket_text = "<|test|>Hello<|end|>"
    encoded = tokenizer.encode(angle_bracket_text)
    assert isinstance(encoded, list), (
        f"encode() with angle brackets should return list for {model_id}"
    )
    assert len(encoded) > 0, (
        f"encode() with angle brackets should be non-empty for {model_id}"
    )


# 已翻譯註解。
@pytest.mark.asyncio
async def test_kimi_tokenizer_specifically():
    """此說明已翻譯為繁體中文。"""
    kimi_models = [
        card for card in await card_cache.list_all() if "kimi" in card.model_id.lower()
    ]

    if not kimi_models:
        pytest.skip("No Kimi models found in MODEL_CARDS")

    model_card = kimi_models[0]
    model_id = model_card.model_id

    model_path = await download_tokenizer_files(model_id)

    # 已翻譯註解。
    if not (model_path / "tokenization_kimi.py").exists():
        pytest.skip("tokenization_kimi.py not found")

    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # 已翻譯註解。
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    assert len(encoded) > 0, "Kimi tokenizer should encode text"
    assert isinstance(decoded, str), "Kimi tokenizer should decode to string"

    # 已翻譯註解。
    assert all(isinstance(t, int) for t in encoded), "Tokens should be integers"

    # 已翻譯註解。
    # 已翻譯註解。
    special_token_text = "<|im_user|>user<|im_middle|>Hello<|im_end|><|im_assistant|>"
    special_encoded = tokenizer.encode(special_token_text)
    assert len(special_encoded) > 0, "Kimi tokenizer should handle special tokens"
    assert all(isinstance(t, int) for t in special_encoded), (
        "Special token encoding should return integers"
    )

    # 已翻譯註解。
    assert eos_token_ids == [163586], "Kimi EOS token should be [163586]"


# 已翻譯註解。
@pytest.mark.asyncio
async def test_glm_tokenizer_specifically():
    """此說明已翻譯為繁體中文。"""

    def contains(card: ModelCard, x: str):
        return x in card.model_id.lower()

    glm_model_cards = [
        card
        for card in await card_cache.list_all()
        if contains(card, "glm")
        and not contains(card, "-5")
        and not contains(card, "4.7")
    ]

    if not glm_model_cards:
        pytest.skip("No GLM models found in MODEL_CARDS")

    model_card = glm_model_cards[0]
    model_id = model_card.model_id

    model_path = await download_tokenizer_files(model_id)

    has_tokenizer = (model_path / "tokenizer.json").exists() or (
        model_path / "tokenizer_config.json"
    ).exists()
    if not has_tokenizer:
        pytest.skip("GLM tokenizer files not found")

    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # 已翻譯註解。
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    assert len(encoded) > 0, "GLM tokenizer should encode text"
    assert isinstance(decoded, str), "GLM tokenizer should decode to string"

    # 已翻譯註解。
    assert eos_token_ids == [
        151336,
        151329,
        151338,
    ], "GLM EOS tokens should be correct"
