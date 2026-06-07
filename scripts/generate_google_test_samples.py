"""Generate test audio samples via Google Cloud TTS.

Usage:
    python -m scripts.generate_google_test_samples
    python -m scripts.generate_google_test_samples --output tests/output

This script uses Google Cloud Text-to-Speech to generate
short Polish audio samples for validation.
"""

from __future__ import annotations

import sys

from preprocess.tts_script_builder import PodcastScript, TTSChunk
from tts import GoogleTTSConfig, generate_podcast_audio_google


def main() -> None:
    """Generate test audio samples via Google Cloud TTS."""
    args = list(sys.argv[1:])

    output_dir = "output/audio/google_samples"

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 1
        i += 1

    config = GoogleTTSConfig(
        language_code="pl-PL",
        voice_name="pl-PL-Wavenet-A",
        output_dir=output_dir,
    )

    print(f"Language: {config.language_code}")
    print(f"Voice: {config.voice_name}")
    print(f"Output directory: {output_dir}")
    print()

    script = PodcastScript(
        title="Google TTS Test Samples",
        chunks=[
            TTSChunk(
                id="sample_1",
                text="Witaj świecie! To jest przykładowy fragment do testowania.",
                section_number=1,
                section_title="Próba 1",
            ),
            TTSChunk(
                id="sample_2",
                text="Generowanie mowy za pomocą Google Cloud TTS jest bardzo wygodne.",
                section_number=2,
                section_title="Próba 2",
            ),
            TTSChunk(
                id="sample_3",
                text="Można generować wiele fragmentów jednocześnie i pobierać ich audio.",
                section_number=3,
                section_title="Próba 3",
            ),
        ],
        metadata={
            "total_sections": 3,
            "total_chunks": 3,
            "total_characters": 230,
        },
    )

    try:
        results = generate_podcast_audio_google(script, config)
        print()
        print("Generated files:")
        for chunk_id, path in results.items():
            print(f"  {chunk_id}: {path}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have Google Cloud credentials configured.")
        sys.exit(1)


if __name__ == "__main__":
    main()
