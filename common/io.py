"""File IO helpers for pipeline stages."""

import json
import re
from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, filepath: Path) -> str:
    ensure_dir(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return str(filepath)


def timestamped_filename(prefix: str, ext: str = "json") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{ext}"


def extract_timestamp_from_filepath(filepath: str, prefix: str = "conversation") -> str | None:
    pattern = rf"{prefix}_(\\d{{8}}_\\d{{6}})"
    match = re.search(pattern, filepath)
    if match:
        return match.group(1)
    return None

