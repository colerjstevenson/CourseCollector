from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import logging


@dataclass
class RunContext:
    repo_root: Path
    target: str
    dry_run: bool
    force: bool
    resume: bool
    log_path: Path
    checkpoint_path: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("course_collector")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mark_stage_done(path: Path, stage_name: str, metadata: dict | None = None) -> None:
    state = load_checkpoint(path)
    done = state.get("done", {})
    done[stage_name] = {
        "completed_at": utc_now_iso(),
        "metadata": metadata or {},
    }
    state["done"] = done
    save_checkpoint(path, state)


def stage_done(path: Path, stage_name: str) -> bool:
    state = load_checkpoint(path)
    done = state.get("done", {})
    return stage_name in done
