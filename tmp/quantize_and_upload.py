#!/usr/bin/env python3
"""
下載 mflux 模型、量化後上傳到 HuggingFace。

使用方式（請從 mflux 專案目錄執行）：
    cd /path/to/mflux
    uv run python /path/to/quantize_and_upload.py --model black-forest-labs/FLUX.1-Kontext-dev
    uv run python /path/to/quantize_and_upload.py --model black-forest-labs/FLUX.1-Kontext-dev --skip-base --skip-8bit
    uv run python /path/to/quantize_and_upload.py --model black-forest-labs/FLUX.1-Kontext-dev --dry-run

需求：
    - Must be run from mflux project directory using `uv run`
    - huggingface_hub installed (add to mflux deps or install separately)
    - HuggingFace authentication: run `huggingface-cli login` or set HF_TOKEN
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mflux.models.flux.variants.txt2img.flux import Flux1


HF_ORG = "exolabs"


def get_model_class(model_name: str) -> type:
    """依模型名稱取得對應的模型類別。"""
    from mflux.models.fibo.variants.txt2img.fibo import FIBO
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
    from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo

    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        return QwenImage
    elif "fibo" in model_name_lower:
        return FIBO
    elif "z-image" in model_name_lower or "zimage" in model_name_lower:
        return ZImageTurbo
    elif "flux2" in model_name_lower or "flux.2" in model_name_lower:
        return Flux2Klein
    else:
        return Flux1


def get_repo_name(model_name: str, bits: int | None) -> str:
    """取得模型變體對應的 HuggingFace repo 名稱。"""
    # 從 HF 路徑擷取 repo 名稱（例如 ...）。
    base_name = model_name.split("/")[-1] if "/" in model_name else model_name
    suffix = f"-{bits}bit" if bits else ""
    return f"{HF_ORG}/{base_name}{suffix}"


def get_local_path(output_dir: Path, model_name: str, bits: int | None) -> Path:
    """取得模型變體的本地儲存路徑。"""
    # 從 HF 路徑擷取 repo 名稱（例如 ...）。
    base_name = model_name.split("/")[-1] if "/" in model_name else model_name
    suffix = f"-{bits}bit" if bits else ""
    return output_dir / f"{base_name}{suffix}"


def copy_source_repo(
    source_repo: str,
    local_path: Path,
    dry_run: bool = False,
) -> None:
    """從來源 repo 複製所有檔案（重現原始 HF 結構）。"""
    print(f"\n{'=' * 60}")
    print(f"Copying full repo from source: {source_repo}")
    print(f"Output path: {local_path}")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN] Would download all files from source repo")
        return

    from huggingface_hub import snapshot_download

    # 下載所有檔案到本地路徑
    snapshot_download(
        repo_id=source_repo,
        local_dir=local_path,
    )

    # 移除根目錄 safetensors 檔（如 flux.1-dev.safetensors 等）
    # 這些檔案與各元件目錄內容重複
    for f in local_path.glob("*.safetensors"):
        print(f"Removing root-level safetensors: {f.name}")
        if not dry_run:
            f.unlink()

    print(f"Source repo copied to {local_path}")


def load_and_save_quantized_model(
    model_name: str,
    bits: int,
    output_path: Path,
    dry_run: bool = False,
) -> None:
    """載入量化模型並以 mflux 格式儲存。"""
    print(f"\n{'=' * 60}")
    print(f"Loading {model_name} with {bits}-bit quantization...")
    print(f"Output path: {output_path}")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN] Would load and save quantized model")
        return

    from mflux.models.common.config.model_config import ModelConfig

    model_class = get_model_class(model_name)
    model_config = ModelConfig.from_name(model_name=model_name, base_model=None)

    model: Flux1 = model_class(
        quantize=bits,
        model_config=model_config,
    )

    print(f"Saving model to {output_path}...")
    model.save_model(str(output_path))
    print(f"Model saved successfully to {output_path}")


def copy_source_metadata(
    source_repo: str,
    local_path: Path,
    dry_run: bool = False,
) -> None:
    """從來源 repo 複製中繼資料檔（LICENSE、README 等），排除 safetensors。"""
    print(f"\n{'=' * 60}")
    print(f"Copying metadata from source repo: {source_repo}")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN] Would download metadata files (excluding *.safetensors)")
        return

    from huggingface_hub import snapshot_download

    # 下載除 safetensors 之外的所有檔案到本地路徑
    snapshot_download(
        repo_id=source_repo,
        local_dir=local_path,
        ignore_patterns=["*.safetensors"],
    )
    print(f"Metadata files copied to {local_path}")


def upload_to_huggingface(
    local_path: Path,
    repo_id: str,
    dry_run: bool = False,
    clean_remote: bool = False,
) -> None:
    """將已儲存模型上傳到 HuggingFace。"""
    print(f"\n{'=' * 60}")
    print(f"Uploading to HuggingFace: {repo_id}")
    print(f"Local path: {local_path}")
    print(f"Clean remote first: {clean_remote}")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN] Would upload to HuggingFace")
        return

    from huggingface_hub import HfApi

    api = HfApi()

    # 若 repo 不存在則建立
    print(f"Creating/verifying repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # 若有要求，先清理遠端 repo（刪除舊 mflux 格式檔案）
    if clean_remote:
        print("Cleaning old mflux-format files from remote...")
        try:
            # mflux 分片檔案樣式：<dir>/<number>.safetensors
            numbered_pattern = re.compile(r".*/\d+\.safetensors$")

            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            for file_path in repo_files:
                # 刪除編號 safetensors（mflux 格式）與 mflux 索引檔
                if numbered_pattern.match(file_path) or file_path.endswith(
                    "/model.safetensors.index.json"
                ):
                    print(f"  Deleting: {file_path}")
                    api.delete_file(
                        path_in_repo=file_path, repo_id=repo_id, repo_type="model"
                    )
        except Exception as e:
            print(f"Warning: Could not clean remote files: {e}")

    # 上傳整個資料夾
    print("Uploading folder contents...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


def clean_local_files(local_path: Path, dry_run: bool = False) -> None:
    """上傳後移除本地模型檔案。"""
    print(f"\nCleaning up: {local_path}")
    if dry_run:
        print("[DRY RUN] Would remove local files")
        return

    if local_path.exists():
        shutil.rmtree(local_path)
        print(f"Removed {local_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download an mflux model, quantize it, and upload to HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all variants (base, 4-bit, 8-bit) for FLUX.1-Kontext-dev
    python tmp/quantize_and_upload.py --model black-forest-labs/FLUX.1-Kontext-dev

    # Only process 4-bit variant
    python tmp/quantize_and_upload.py --model black-forest-labs/FLUX.1-Kontext-dev --skip-base --skip-8bit

    # Save locally without uploading
    python tmp/quantize_and_upload.py --model black-forest-labs/FLUX.1-Kontext-dev --skip-upload

    # Preview what would happen
    python tmp/quantize_and_upload.py --model black-forest-labs/FLUX.1-Kontext-dev --dry-run
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="HuggingFace model path (e.g., black-forest-labs/FLUX.1-Kontext-dev)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./tmp/models"),
        help="Local directory to save models (default: ./tmp/models)",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model (no quantization)",
    )
    parser.add_argument(
        "--skip-4bit",
        action="store_true",
        help="Skip 4-bit quantized model",
    )
    parser.add_argument(
        "--skip-8bit",
        action="store_true",
        help="Skip 8-bit quantized model",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading/processing, only do upload/clean operations",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only save locally, don't upload to HuggingFace",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove local files after upload",
    )
    parser.add_argument(
        "--clean-remote",
        action="store_true",
        help="Delete old mflux-format files from remote repo before uploading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing",
    )

    args = parser.parse_args()

    # 決定要處理哪些變體
    variants: list[int | None] = []
    if not args.skip_base:
        variants.append(None)  # 基礎模型（不量化）
    if not args.skip_4bit:
        variants.append(4)
    if not args.skip_8bit:
        variants.append(8)

    if not variants:
        print("Error: All variants skipped. Nothing to do.")
        return 1

    # 建立輸出目錄
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Variants to process: {['base' if v is None else f'{v}-bit' for v in variants]}"
    )
    print(f"Upload to HuggingFace: {not args.skip_upload}")
    print(f"Clean after upload: {args.clean}")
    if args.dry_run:
        print("\n*** DRY RUN MODE - No actual changes will be made ***")

    # 逐一處理每個變體
    for bits in variants:
        local_path = get_local_path(args.output_dir, args.model, bits)
        repo_id = get_repo_name(args.model, bits)

        if not args.skip_download:
            if bits is None:
                # 基礎模型：複製原始 HF repo 結構（不做 mflux 轉換）
                copy_source_repo(
                    source_repo=args.model,
                    local_path=local_path,
                    dry_run=args.dry_run,
                )
            else:
                # 量化模型：載入、量化並以 mflux 儲存
                load_and_save_quantized_model(
                    model_name=args.model,
                    bits=bits,
                    output_path=local_path,
                    dry_run=args.dry_run,
                )

                # 從來源 repo 複製中繼資料（LICENSE、README 等）
                copy_source_metadata(
                    source_repo=args.model,
                    local_path=local_path,
                    dry_run=args.dry_run,
                )

        # 上傳
        if not args.skip_upload:
            upload_to_huggingface(
                local_path=local_path,
                repo_id=repo_id,
                dry_run=args.dry_run,
                clean_remote=args.clean_remote,
            )

            # 若有要求則清理
            if args.clean:
                clean_local_files(local_path, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
