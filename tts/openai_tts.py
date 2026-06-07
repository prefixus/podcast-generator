"""Text-to-speech generation using OpenAI API (tts-1 / tts-1-hd).

Takes TTSChunk objects from the preprocess pipeline and generates
high-quality audio files via the OpenAI Speech API. Audio files are
saved as WAV for later merging into full podcast chapters.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from preprocess.tts_script_builder import PodcastScript, TTSChunk


@dataclass
class OpenAITTSConfig:
    """Configuration for OpenAI TTS generation."""

    model: str = "tts-1"
    voice: str = "alloy"
    speed: float = 1.0
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "wav"
    output_dir: str | Path = "output/audio"
    api_key: str | None = None
    base_url: str | None = None
    # Maximum characters per API call (safety limit)
    max_chars_per_call: int = 4096

    # Supported voices for the tts-1 / tts-1-hd models
    SUPPORTED_VOICES: set[str] = field(
        default_factory=lambda: {"alloy", "echo", "fable", "onyx", "nova", "shimmer"},
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.voice not in self.SUPPORTED_VOICES:
            raise ValueError(f"Unsupported voice '{self.voice}'. Supported: {sorted(self.SUPPORTED_VOICES)}")
        if not 0.25 <= self.speed <= 2.0:
            raise ValueError("Speed must be between 0.25 and 2.0")

    def _get_api_key(self) -> str:
        """Resolve API key: explicit > env var > default."""
        if self.api_key:
            return self.api_key
        import os

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "No OpenAI API key provided. Set OPENAI_API_KEY env var or pass api_key to OpenAITTSConfig."
            )
        return key


def _build_client(config: OpenAITTSConfig) -> OpenAI:
    """Create an OpenAI client, optionally using a custom base URL."""
    kwargs: dict[str, Any] = {"api_key": config._get_api_key()}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def _generate_wav_from_bytes(raw_audio: bytes) -> bytes:
    """Wrap raw Opus audio bytes into a minimal WAV container.

    OpenAI returns raw Opus-encoded data. We wrap it in a WAV header
    so downstream tools (ffmpeg, audacity) can play it without
    additional transcoding.
    """
    import io

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        # OpenAI tts-1 outputs 24kHz Opus; write as PCM placeholder
        # The raw bytes are written as-is; most players handle this.
        wav_file.writeframes(raw_audio)
    return wav_buffer.getvalue()


def save_audio_file(audio_bytes: bytes, output_path: str | Path) -> Path:
    """Save audio bytes to disk, creating directories as needed.

    If the bytes are raw Opus (not WAV), wraps them in a WAV container.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Detect if we already have a WAV file
    if output.suffix.lower() == ".wav":
        # Check WAV header magic
        if audio_bytes[:4] == b"RIFF":
            output.write_bytes(audio_bytes)
            return output

    # Wrap in WAV container
    wav_data = _generate_wav_from_bytes(audio_bytes)
    output.write_bytes(wav_data)
    return output


def generate_audio_chunk(
    chunk: TTSChunk,
    config: OpenAITTSConfig,
) -> bytes:
    """Send a single TTSChunk text to OpenAI and return audio bytes.

    If the text exceeds max_chars_per_call, the text is split into
    smaller segments, each sent separately, and the results are
    concatenated.
    """
    client = _build_client(config)
    text = chunk.text

    # Handle oversized chunks by splitting on sentence boundaries
    import re

    if len(text) > config.max_chars_per_call:
        segments = re.split(r"(?<=[.!?])\s+", text)
        all_audio: list[bytes] = []
        current_segment: list[str] = []
        current_len = 0

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if current_len + len(seg) > config.max_chars_per_call and current_segment:
                all_audio.append(_call_api(client, config, " ".join(current_segment)))
                current_segment = [seg]
                current_len = len(seg)
            else:
                current_segment.append(seg)
                current_len += len(seg)

        if current_segment:
            all_audio.append(_call_api(client, config, " ".join(current_segment)))

        # Concatenate audio bytes (works for WAV containers)
        return b"".join(all_audio)

    return _call_api(client, config, text)


def _call_api(
    client: OpenAI,
    config: OpenAITTSConfig,
    text: str,
) -> bytes:
    """Make a single OpenAI Speech API call and return raw audio bytes."""
    response = client.audio.speech.create(
        model=config.model,
        voice=config.voice,
        input=text,
        response_format=config.response_format,
        speed=config.speed,
    )
    return response.content


def generate_podcast_audio(
    script: PodcastScript,
    config: OpenAITTSConfig | None = None,
) -> dict[str, str]:
    """Generate audio for every chunk in a PodcastScript.

    Returns a mapping of chunk id → saved file path.
    """
    if config is None:
        config = OpenAITTSConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}

    for chunk in script.chunks:
        print(f"Generating audio for chunk: {chunk.id} ({len(chunk.text)} chars)")
        audio_bytes = generate_audio_chunk(chunk, config)
        audio_path = save_audio_file(
            audio_bytes,
            output_dir / f"{chunk.id}.wav",
        )
        results[chunk.id] = str(audio_path)

    # Save manifest for later merging
    manifest_path = output_dir / "manifest.json"
    manifest_data = {
        "title": script.title,
        "metadata": script.metadata,
        "chunks": [
            {
                "id": chunk.id,
                "text": chunk.text,
                "section_number": chunk.section_number,
                "section_title": chunk.section_title,
                "is_intro": chunk.is_intro,
                "is_outro": chunk.is_outro,
                "is_transition": chunk.is_transition,
                "audio_file": results[chunk.id],
            }
            for chunk in script.chunks
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Manifest saved to: {manifest_path}")
    return results


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a previously generated manifest for chapter merging."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
