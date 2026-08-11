from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


DEFAULT_DATASET_ID = "colerjstevenson/GolfGulf"
DEFAULT_INCLUDE = "**/*.json,**/*.geojson,**/*.csv"


def _read_token(token_file: Path, env_var: str = "HF_TOKEN") -> str:
    token = os.getenv(env_var, "").strip()
    if token:
        return token

    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()

    raise SystemExit(
        "Missing Hugging Face token. Set HF_TOKEN or provide a local token file."
    )


def _parse_include_csv(include_csv: str) -> list[str]:
    patterns = [p.strip() for p in include_csv.split(",") if p.strip()]
    if not patterns:
        return ["**/*"]
    return patterns


def _collect_files(source_dir: Path, include_patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(source_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in include_patterns):
            files.append(path)

    return sorted(files)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(
    dataset_id: str,
    source_dir: Path,
    files: Iterable[Path],
    dataset_prefix: str,
) -> dict:
    normalized_prefix = dataset_prefix.strip("/")
    items = []
    for file_path in files:
        rel = file_path.relative_to(source_dir).as_posix()
        repo_path = f"{normalized_prefix}/{rel}" if normalized_prefix else rel
        items.append(
            {
                "path": repo_path,
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
        )

    return {
        "dataset_id": dataset_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(items),
        "files": items,
    }


def _delete_removed_files(
    api: HfApi,
    dataset_id: str,
    token: str,
    local_files: Iterable[Path],
    source_dir: Path,
    dataset_prefix: str,
) -> int:
    prefix = dataset_prefix.strip("/")
    local_repo_paths = set()
    for file_path in local_files:
        rel = file_path.relative_to(source_dir).as_posix()
        local_repo_paths.add(f"{prefix}/{rel}" if prefix else rel)

    local_repo_paths.add("manifest.json")

    existing_repo_files = api.list_repo_files(
        repo_id=dataset_id,
        repo_type="dataset",
        token=token,
    )

    to_delete: list[str] = []
    for path_in_repo in existing_repo_files:
        if path_in_repo == "README.md":
            continue
        if prefix:
            if not path_in_repo.startswith(prefix + "/"):
                continue
        if path_in_repo not in local_repo_paths:
            to_delete.append(path_in_repo)

    for path_in_repo in to_delete:
        api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=dataset_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Remove stale file: {path_in_repo}",
        )

    return len(to_delete)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish local dataset files to a Hugging Face dataset repository."
    )
    parser.add_argument("--dataset", default=os.getenv("HF_DATASET_ID", DEFAULT_DATASET_ID))
    parser.add_argument("--source", default="data")
    parser.add_argument("--token-file", default="token")
    parser.add_argument("--dataset-prefix", default="")
    parser.add_argument("--include-csv", default=DEFAULT_INCLUDE)
    parser.add_argument("--delete-removed", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    token_file = Path(args.token_file).resolve()
    include_patterns = _parse_include_csv(args.include_csv)

    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    token = _read_token(token_file)
    dataset_id = args.dataset.strip() or DEFAULT_DATASET_ID

    files = _collect_files(source_dir, include_patterns)
    if not files:
        raise SystemExit(
            f"No files matched include patterns in {source_dir}: {include_patterns}"
        )

    api = HfApi()
    api.create_repo(
        repo_id=dataset_id,
        repo_type="dataset",
        token=token,
        private=False,
        exist_ok=True,
    )

    prefix = args.dataset_prefix.strip("/")
    uploaded = 0
    for file_path in files:
        rel = file_path.relative_to(source_dir).as_posix()
        path_in_repo = f"{prefix}/{rel}" if prefix else rel
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=dataset_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload {path_in_repo}",
        )
        uploaded += 1

    manifest = _build_manifest(dataset_id, source_dir, files, prefix)
    manifest_text = json.dumps(manifest, indent=2)
    api.upload_file(
        path_or_fileobj=manifest_text.encode("utf-8"),
        path_in_repo="manifest.json",
        repo_id=dataset_id,
        repo_type="dataset",
        token=token,
        commit_message="Update manifest.json",
    )

    deleted = 0
    if args.delete_removed:
        deleted = _delete_removed_files(
            api=api,
            dataset_id=dataset_id,
            token=token,
            local_files=files,
            source_dir=source_dir,
            dataset_prefix=prefix,
        )

    print(f"Dataset: {dataset_id}")
    print(f"Uploaded files: {uploaded}")
    print(f"Deleted stale files: {deleted}")
    print("Manifest uploaded: manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())