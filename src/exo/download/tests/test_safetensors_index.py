"""此說明已翻譯為繁體中文。"""

from exo.shared.types.worker.downloads import ModelSafetensorsIndex


def test_safetensors_index_missing_total_size():
    """此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """
    raw = '{"metadata": {"quantization_level": "4", "mflux_version": "0.3.0"}, "weight_map": {"layer.safetensors": "layer.safetensors"}}'
    index = ModelSafetensorsIndex.model_validate_json(raw)
    assert index.metadata is not None
    assert index.metadata.total_size is None
    assert index.weight_map == {"layer.safetensors": "layer.safetensors"}


def test_safetensors_index_valid_total_size():
    """此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。"""
    raw = '{"metadata": {"total_size": 12345}, "weight_map": {"a.safetensors": "a.safetensors"}}'
    index = ModelSafetensorsIndex.model_validate_json(raw)
    assert index.metadata is not None
    assert index.metadata.total_size == 12345


def test_safetensors_index_null_metadata():
    """此說明已翻譯為繁體中文。"""
    raw = '{"metadata": null, "weight_map": {"a.safetensors": "a.safetensors"}}'
    index = ModelSafetensorsIndex.model_validate_json(raw)
    assert index.metadata is None
