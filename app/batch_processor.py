from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from .transcribe_api import build_client, transcribe_audio
from .utils import ensure_output_dir, get_output_path, list_audio_files


def process_batch(input_dir: Path, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    client = build_client()

    audio_files = list_audio_files(input_dir)
    if not audio_files:
        logging.info("No supported audio files found in %s", input_dir)
        return

    for audio_file in audio_files:
        output_path = get_output_path(audio_file, output_dir)
        # Skip if a transcript already exists.
        if output_path.exists():
            logging.info("Skipping (already transcribed): %s", audio_file)
            continue

        try:
            transcribe_audio(audio_file, output_dir, client)
        except Exception as exc:
            logging.error("Failed to transcribe %s: %s", audio_file, exc)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    input_dir = Path("input")
    output_dir = Path("output")

    process_batch(input_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
