#!/usr/bin/env python3
"""Download Qwen3-14B model for local YMCA agent inference.

Usage:
    python scripts/download_qwen.py
    python scripts/download_qwen.py --model Qwen/Qwen3-14B --output /path/to/models/qwen/Qwen3-14B
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-14B"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "models" / "qwen" / "Qwen3-14B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen model from Hugging Face Hub.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Local output directory")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model} -> {args.output}")
    snapshot_download(
        repo_id=args.model,
        local_dir=str(args.output),
        local_dir_use_symlinks=False,
    )
    print(f"Done. Model at: {args.output}")


if __name__ == "__main__":
    main()
