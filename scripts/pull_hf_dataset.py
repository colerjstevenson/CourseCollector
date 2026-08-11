from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_DATASET_ID = "colerjstevenson/GolfGulf"
DEFAULT_INCLUDE = "**/*.json,**/*.geojson,**/*.csv,manifest.json"


def _optional_token(token_file: Path, env_var: str = "HF_TOKEN") -> str | None:
    token = os.getenv(env_var, "").strip()
    if token:
        return token
    if token_file.exists():
        value = token_file.read_text(encoding="utf-8").strip()
        return value or None
    return None


def _parse_include_csv(include_csv: str) -> list[str]:
    patterns = [p.strip() for p in include_csv.split(",") if p.strip()]
    if not patterns:
        return ["**/*"]
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull a Hugging Face dataset snapshot into a local directory."
    )
    parser.add_argument("--dataset", default=os.getenv("HF_DATASET_ID", DEFAULT_DATASET_ID))
    parser.add_argument("--output", default="maps/data")
    parser.add_argument("--token-file", default="token")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--include-csv", default=DEFAULT_INCLUDE)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    dataset_id = args.dataset.strip() or DEFAULT_DATASET_ID
    output_dir = Path(args.output).resolve()
    token_file = Path(args.token_file).resolve()
    allow_patterns = _parse_include_csv(args.include_csv)
    token = _optional_token(token_file)

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=args.revision,
        token=token,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    print(f"Dataset: {dataset_id}")
    print(f"Output: {output_dir}")
    print(f"Included patterns: {allow_patterns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())