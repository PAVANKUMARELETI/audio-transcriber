from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from .utils import ensure_output_dir, get_output_path, validate_audio_file


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def transcribe_audio(input_path: Path, output_dir: Path, client: OpenAI) -> Path:
    validate_audio_file(input_path)
    ensure_output_dir(output_dir)

    output_path = get_output_path(input_path, output_dir)
    logging.info("Transcribing %s", input_path)

    # Call Whisper API and request plain text output.
    with input_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    output_path.write_text(response, encoding="utf-8")
    logging.info("Saved transcript to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a single audio file using Whisper API.")
    parser.add_argument("file", help="Path to an audio file (mp3, wav, m4a, mp4)")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for transcript text files (default: output)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    input_path = Path(args.file)
    output_dir = Path(args.output_dir)

    try:
        client = build_client()
        transcribe_audio(input_path, output_dir, client)
        return 0
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return 2
    except ValueError as exc:
        logging.error(str(exc))
        return 3
    except (APIError, APIConnectionError, RateLimitError) as exc:
        logging.error("OpenAI API error: %s", exc)
        return 4
    except EnvironmentError as exc:
        logging.error(str(exc))
        return 5
    except Exception as exc:  # pragma: no cover - safety net
        logging.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
