"""Text-to-speech module using OpenAI API.

Generates audio files from TTS-ready chunks produced by the
preprocess pipeline (pdf_extractor → tts_script_builder).
"""

from tts.openai_tts import (
    OpenAITTSConfig,
    generate_audio_chunk,
    generate_podcast_audio,
    load_manifest,
    save_audio_file,
)

__all__ = [
    "OpenAITTSConfig",
    "generate_audio_chunk",
    "generate_podcast_audio",
    "load_manifest",
    "save_audio_file",
]
