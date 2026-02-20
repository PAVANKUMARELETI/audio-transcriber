from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4"}


def is_supported_audio(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def validate_audio_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Not a file: {file_path}")
    if not is_supported_audio(file_path):
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported format: {file_path.suffix}. Supported: {supported}")


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def get_output_path(input_path: Path, output_dir: Path) -> Path:
    safe_name = input_path.stem.replace(" ", "_")
    return output_dir / f"{safe_name}.txt"


def list_audio_files(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        logging.warning("Input folder does not exist: %s", input_dir)
        return []
    return sorted(p for p in input_dir.iterdir() if p.is_file() and is_supported_audio(p))
