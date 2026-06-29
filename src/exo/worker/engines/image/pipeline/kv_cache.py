import mlx.core as mx


class ImagePatchKVCache:
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        image_seq_len: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.image_seq_len = image_seq_len
        self.head_dim = head_dim
        self._dtype = dtype

        self.key_cache = mx.zeros(
            (batch_size, num_heads, image_seq_len, head_dim), dtype=dtype
        )
        self.value_cache = mx.zeros(
            (batch_size, num_heads, image_seq_len, head_dim), dtype=dtype
        )

    def update_image_patch(
        self, patch_start: int, patch_end: int, key: mx.array, value: mx.array
    ) -> None:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        self.key_cache[:, :, patch_start:patch_end, :] = key
        self.value_cache[:, :, patch_start:patch_end, :] = value

    def get_full_kv(
        self, text_key: mx.array, text_value: mx.array
    ) -> tuple[mx.array, mx.array]:
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
            此說明已翻譯為繁體中文。
        """
        full_key = mx.concatenate([text_key, self.key_cache], axis=2)
        full_value = mx.concatenate([text_value, self.value_cache], axis=2)
        return full_key, full_value

    def reset(self) -> None:
        """此說明已翻譯為繁體中文。"""
        self.key_cache = mx.zeros(
            (self.batch_size, self.num_heads, self.image_seq_len, self.head_dim),
            dtype=self._dtype,
        )
        self.value_cache = mx.zeros(
            (self.batch_size, self.num_heads, self.image_seq_len, self.head_dim),
            dtype=self._dtype,
        )
