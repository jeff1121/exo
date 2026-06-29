"""
為 EXO 產生推論模型卡。
使用方式：
    uv run tmp/gen_card.py mlx-community/my_cool_model-8bit [repo-id/model-id-2] [...]

Model Cards 仍需清理模型家族與量化相關資料
"""

import sys

import anyio

from exo.shared.models.model_cards import ModelCard, ModelId


async def main():
    if len(sys.argv) == 1:
        print(f"USAGE: {sys.argv[0]} repo-id/model-id-1 [repo-id/model-id-2] [...]")
        quit(1)
    print("提醒：Model Cards 仍需清理模型家族與量化相關資料")
    for arg in sys.argv[1:]:
        mid = ModelId(arg)
        mc = await ModelCard.fetch_from_hf(mid)
        await mc.save(
            anyio.Path(__file__).parent.parent
            / "resources"
            / "inference_model_cards"
            / (mid.normalize() + ".toml")
        )


if __name__ == "__main__":
    anyio.run(main)
