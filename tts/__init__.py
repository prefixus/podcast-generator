"""Text-to-speech module supporting multiple backends.

Generates audio files from TTS-ready chunks produced by the
preprocess pipeline (pdf_extractor → tts_script_builder).

Supported backends:
  - OpenAI Speech API (default)
  - Local FastAPI server (e.g. Higgs Audio v3)
"""

from tts.local_tts import (
    LocalTTSConfig,
    generate_podcast_audio as generate_podcast_audio_local,
    generate_test_samples,
    load_manifest as load_manifest_local,
    save_audio_file as save_audio_file_local,
)
from tts.openai_tts import (
    OpenAITTSConfig,
    generate_audio_chunk,
    generate_podcast_audio,
    load_manifest,
    save_audio_file,
)

__all__ = [
    # OpenAI backend
    "OpenAITTSConfig",
    "generate_audio_chunk",
    "generate_podcast_audio",
    "load_manifest",
    "save_audio_file",
    # Local TTS backend
    "LocalTTSConfig",
    "generate_podcast_audio_local",
    "generate_test_samples",
    "load_manifest_local",
    "save_audio_file_local",
]
