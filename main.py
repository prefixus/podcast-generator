"""Podcast Creator – PDF-to-TTS full pipeline.

Extracts structured content from PDF documents, builds TTS-ready
scripts, and generates audio files via the OpenAI Speech API.

Usage:
    # Preprocess only (script generation):
    python main.py

    # Full pipeline (script + audio generation):
    python main.py --tts

    # Custom PDF with TTS:
    python main.py path/to/file.pdf --tts --voice nova --model tts-1-hd
"""

from __future__ import annotations

import sys
from pathlib import Path

from preprocess.pdf_extractor import extract_and_parse
from preprocess.tts_script_builder import (
    PodcastScript,
    build_podcast_script,
    save_script_json,
    save_script_text,
)
from tts import OpenAITTSConfig, generate_podcast_audio


def process_pdf(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    generate_audio: bool = False,
    tts_config: OpenAITTSConfig | None = None,
) -> PodcastScript:
    """Full pipeline: PDF → structured sections → TTS-ready script.

    Optionally generates audio files via OpenAI Speech API when
    ``generate_audio=True``.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract and parse PDF
    doc = extract_and_parse(pdf_path)

    # Step 2: Build podcast script
    script = build_podcast_script(doc)

    # Step 3: Save outputs
    stem = pdf_path.stem
    save_script_json(script, output_dir / f"{stem}_script.json")
    save_script_text(script, output_dir / f"{stem}_script.txt")

    # Step 4 (optional): Generate audio via OpenAI TTS
    if generate_audio:
        if tts_config is None:
            tts_config = OpenAITTSConfig(output_dir=output_dir / "audio")
        audio_map = generate_podcast_audio(script, tts_config)
        print(f"Generated {len(audio_map)} audio files")

    # Print summary
    print(f"Document: {doc.title}")
    print(f"Sections extracted: {script.metadata['total_sections']}")
    print(f"TTS chunks: {script.metadata['total_chunks']}")
    print(f"Total characters: {script.metadata['total_characters']}")
    print(f"Output saved to: {output_dir}")

    return script


def main() -> None:
    """Entry point: process the example PDF (optionally generate audio)."""
    args = list(sys.argv[1:])

    # Simple argument parsing
    generate_audio = "--tts" in args or "--audio" in args
    tts_voice = "alloy"
    tts_model = "tts-1"
    output_dir = "output"
    pdf_file = "example-data/Seksuologia_opracowane_tezy.pdf"

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--tts", "--audio"):
            pass  # handled above
        elif arg == "--voice" and i + 1 < len(args):
            tts_voice = args[i + 1]
            i += 1
        elif arg == "--model" and i + 1 < len(args):
            tts_model = args[i + 1]
            i += 1
        elif arg == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 1
        elif not arg.startswith("-"):
            pdf_file = arg
        i += 1

    tts_config = OpenAITTSConfig(
        voice=tts_voice,
        model=tts_model,
        output_dir=Path(output_dir) / "audio",
    )

    process_pdf(pdf_file, output_dir, generate_audio=generate_audio, tts_config=tts_config)


if __name__ == "__main__":
    main()
