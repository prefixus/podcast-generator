"""Podcast Creator – PDF-to-TTS full pipeline.

Extracts structured content from PDF documents, builds TTS-ready
scripts, and generates audio files via supported TTS backends:

  - OpenAI Speech API (default)
  - Local FastAPI server (e.g. Higgs Audio v3)
  - Google Cloud Text-to-Speech

Usage:
    # Preprocess only (script generation):
    python main.py

    # Full pipeline with OpenAI TTS:
    python main.py --tts

    # Full pipeline with local TTS server:
    python main.py --tts --local-tts --tts-host 127.0.0.1 --tts-port 8000

    # Full pipeline with Google Cloud TTS:
    python main.py --tts --google-tts --google-voice pl-PL-Wavenet-A

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
from tts import (
    GoogleTTSConfig,
    LocalTTSConfig,
    OpenAITTSConfig,
    generate_podcast_audio,
    generate_podcast_audio_google,
    generate_podcast_audio_local,
)


def process_pdf(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    generate_audio: bool = False,
    use_local_tts: bool = False,
    use_google_tts: bool = False,
    tts_config: OpenAITTSConfig | None = None,
    local_tts_config: LocalTTSConfig | None = None,
    google_tts_config: GoogleTTSConfig | None = None,
) -> PodcastScript:
    """Full pipeline: PDF → structured sections → TTS-ready script.

    Optionally generates audio files via OpenAI Speech API, a local
    FastAPI TTS server, or Google Cloud TTS when ``generate_audio=True``.
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

    # Step 4 (optional): Generate audio
    if generate_audio:
        if use_local_tts:
            if local_tts_config is None:
                local_tts_config = LocalTTSConfig(output_dir=output_dir / "audio")
            audio_map = generate_podcast_audio_local(script, local_tts_config)
            print(f"Generated {len(audio_map)} audio files via local TTS")
        elif use_google_tts:
            if google_tts_config is None:
                google_tts_config = GoogleTTSConfig(output_dir=output_dir / "audio")
            audio_map = generate_podcast_audio_google(script, google_tts_config)
            print(f"Generated {len(audio_map)} audio files via Google Cloud TTS")
        else:
            if tts_config is None:
                tts_config = OpenAITTSConfig(output_dir=output_dir / "audio")
            audio_map = generate_podcast_audio(script, tts_config)
            print(f"Generated {len(audio_map)} audio files via OpenAI TTS")

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
    use_local_tts = "--local-tts" in args
    use_google_tts = "--google-tts" in args
    tts_voice = "alloy"
    tts_model = "tts-1"
    output_dir = "output"
    pdf_file = "example-data/Seksuologia_opracowane_tezy.pdf"
    tts_host = "127.0.0.1"
    tts_port = 8000
    tts_language = "pl"
    google_voice = "pl-PL-Wavenet-A"
    google_language = "pl-PL"
    google_credentials = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--tts", "--audio"):
            pass  # handled above
        elif arg == "--local-tts":
            use_local_tts = True
        elif arg == "--google-tts":
            use_google_tts = True
        elif arg == "--voice" and i + 1 < len(args):
            tts_voice = args[i + 1]
            i += 1
        elif arg == "--model" and i + 1 < len(args):
            tts_model = args[i + 1]
            i += 1
        elif arg == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 1
        elif arg == "--tts-host" and i + 1 < len(args):
            tts_host = args[i + 1]
            i += 1
        elif arg == "--tts-port" and i + 1 < len(args):
            try:
                tts_port = int(args[i + 1])
            except ValueError:
                print(f"Warning: Invalid port '{args[i + 1]}', using default {tts_port}")
            i += 1
        elif arg == "--tts-language" and i + 1 < len(args):
            tts_language = args[i + 1]
            i += 1
        elif arg == "--google-voice" and i + 1 < len(args):
            google_voice = args[i + 1]
            i += 1
        elif arg == "--google-language" and i + 1 < len(args):
            google_language = args[i + 1]
            i += 1
        elif arg == "--google-credentials" and i + 1 < len(args):
            google_credentials = args[i + 1]
            i += 1
        elif not arg.startswith("-"):
            pdf_file = arg
        i += 1

    if use_local_tts:
        local_tts_config = LocalTTSConfig(
            host=tts_host,
            port=tts_port,
            language=tts_language,
            output_dir=Path(output_dir) / "audio",
        )
        process_pdf(
            pdf_file,
            output_dir,
            generate_audio=generate_audio,
            use_local_tts=True,
            local_tts_config=local_tts_config,
        )
    elif use_google_tts:
        google_tts_config = GoogleTTSConfig(
            voice_name=google_voice,
            language_code=google_language,
            credentials_file=google_credentials,
            output_dir=Path(output_dir) / "audio",
        )
        process_pdf(
            pdf_file,
            output_dir,
            generate_audio=generate_audio,
            use_google_tts=True,
            google_tts_config=google_tts_config,
        )
    else:
        tts_config = OpenAITTSConfig(
            voice=tts_voice,
            model=tts_model,
            output_dir=Path(output_dir) / "audio",
        )
        process_pdf(
            pdf_file,
            output_dir,
            generate_audio=generate_audio,
            use_local_tts=False,
            tts_config=tts_config,
        )


if __name__ == "__main__":
    main()
