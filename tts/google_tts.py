"""Text-to-speech generation using Google Cloud Text-to-Speech API.

Takes TTSChunk objects from the preprocess pipeline and generates
audio files via Google Cloud TTS. Audio files are saved as WAV.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.cloud import texttospeech

from preprocess.tts_script_builder import PodcastScript, TTSChunk


@dataclass
class GoogleTTSConfig:
    """Configuration for Google Cloud TTS generation."""

    language_code: str = "pl-PL"
    voice_name: str = "pl-PL-Wavenet-A"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    audio_encoding: str = "LINEAR16"
    output_dir: str | Path = "output/audio"
    credentials_file: str | None = None
    # Maximum characters per API call (safety limit)
    max_chars_per_call: int = 4096

    def __post_init__(self) -> None:
        if not 0.25 <= self.speaking_rate <= 4.0:
            raise ValueError("speaking_rate must be between 0.25 and 4.0")
        if not -20.0 <= self.pitch <= 20.0:
            raise ValueError("pitch must be between -20.0 and 20.0")


def save_audio_file(audio_bytes: bytes, output_path: str | Path) -> Path:
    """Save audio bytes to disk, creating directories as needed."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(audio_bytes)
    return output


def generate_audio_chunk(
    chunk: TTSChunk,
    config: GoogleTTSConfig,
) -> bytes:
    """Send a single TTSChunk text to Google Cloud TTS and return audio bytes.

    If the text exceeds max_chars_per_call, the text is split into
    smaller segments, each sent separately, and the results are
    concatenated.
    """
    kwargs: dict[str, Any] = {}
    if config.credentials_file:
        kwargs["credentials_file"] = config.credentials_file
    client = texttospeech.TextToSpeechClient(**kwargs)  # type: ignore[arg-type]

    text = chunk.text

    # Handle oversized chunks by splitting on sentence boundaries
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

        return b"".join(all_audio)

    return _call_api(client, config, text)


def _call_api(
    client: texttospeech.TextToSpeechClient,
    config: GoogleTTSConfig,
    text: str,
) -> bytes:
    """Make a single Google Cloud TTS API call and return audio bytes."""
    input_text = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=config.language_code,
        name=config.voice_name,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=config.speaking_rate,
        pitch=config.pitch,
    )
    request = texttospeech.SynthesizeSpeechRequest(
        input=input_text,
        voice=voice_params,
        audio_config=audio_config,
    )
    response = client.synthesize_speech(request=request)
    return response.audio_content


def generate_podcast_audio(
    script: PodcastScript,
    config: GoogleTTSConfig | None = None,
) -> dict[str, str]:
    """Generate audio for every chunk in a PodcastScript.

    Returns a mapping of chunk id → saved file path.
    """
    if config is None:
        config = GoogleTTSConfig()

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
