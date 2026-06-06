"""Tests for the OpenAI TTS module."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.tts_script_builder import PodcastScript, TTSChunk
from tts import (
    OpenAITTSConfig,
    generate_podcast_audio,
    load_manifest,
    save_audio_file,
)

# ── Config tests ──────────────────────────────────────────────


class TestOpenAITTSConfig:
    """Tests for OpenAITTSConfig validation."""

    def test_default_config(self) -> None:
        cfg = OpenAITTSConfig()
        assert cfg.model == "tts-1"
        assert cfg.voice == "alloy"
        assert cfg.speed == 1.0
        assert cfg.response_format == "wav"
        assert cfg.max_chars_per_call == 4096

    def test_invalid_voice_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported voice"):
            OpenAITTSConfig(voice="invalid_voice")

    def test_invalid_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="Speed must be between"):
            OpenAITTSConfig(speed=3.0)

    def test_valid_speeds(self) -> None:
        for s in (0.25, 0.5, 1.0, 1.5, 2.0):
            OpenAITTSConfig(speed=s)  # no exception

    def test_custom_output_dir(self) -> None:
        cfg = OpenAITTSConfig(output_dir="custom/audio/path")
        assert cfg.output_dir == "custom/audio/path"

    def test_custom_base_url(self) -> None:
        cfg = OpenAITTSConfig(base_url="https://custom.api/v1")
        assert cfg.base_url == "https://custom.api/v1"

    def test_no_api_key_no_env(self) -> None:
        """Should raise when neither api_key nor env var is set."""
        cfg = OpenAITTSConfig(api_key=None)
        # Unset the env var temporarily
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(RuntimeError, match="No OpenAI API key"):
                cfg._get_api_key()
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


# ── save_audio_file tests ─────────────────────────────────────


class TestSaveAudioFile:
    """Tests for save_audio_file helper."""

    def test_saves_wav_bytes(self, tmp_path: Path) -> None:
        fake_wav = b"RIFF....WAV"  # minimal RIFF header
        result = save_audio_file(fake_wav, tmp_path / "test.wav")
        assert result.exists()
        assert result.read_bytes() == fake_wav

    def test_wraps_raw_opus_in_wav(self, tmp_path: Path) -> None:
        """Non-WAV bytes get wrapped in a WAV container."""
        fake_opus = b"this is raw opus data"
        result = save_audio_file(fake_opus, tmp_path / "test.wav")
        assert result.exists()
        # Should NOT be the raw bytes (wrapped in WAV header)
        assert result.read_bytes() != fake_opus

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c.wav"
        result = save_audio_file(b"RIFF....WAV", nested)
        assert result.exists()


# ── Integration tests (mocked API) ────────────────────────────


class TestGeneratePodcastAudio:
    """End-to-end test with mocked OpenAI API."""

    @patch("tts.openai_tts.OpenAI")
    def test_generates_audio_for_chunks(self, mock_openai: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock the speech.create response
        mock_response = MagicMock()
        mock_response.content = b"fake_audio_data"
        mock_client.audio.speech.create.return_value = mock_response

        script = PodcastScript(
            title="Test Episode",
            chunks=[
                TTSChunk(
                    id="intro",
                    text="Witajcie w odcinku testowym.",
                    section_number=0,
                    section_title="Wstęp",
                    is_intro=True,
                ),
                TTSChunk(
                    id="chunk_1_0",
                    text="To jest przykładowy tekst do syntezy mowy.",
                    section_number=1,
                    section_title="Testowy temat",
                ),
            ],
        )

        cfg = OpenAITTSConfig(
            api_key="sk-fake-key-for-testing",
            output_dir="output/audio/test_mock",
        )

        result = generate_podcast_audio(script, cfg)

        assert len(result) == 2
        assert "intro" in result
        assert "chunk_1_0" in result

        # Check manifest was saved
        manifest_path = Path(cfg.output_dir) / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["title"] == "Test Episode"
        assert len(manifest["chunks"]) == 2

    @patch("tts.openai_tts.OpenAI")
    def test_handles_oversized_chunk(self, mock_openai: MagicMock) -> None:
        """Oversized chunks are split and multiple API calls are made."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = b"fake"
        mock_client.audio.speech.create.return_value = mock_response

        long_text = "To jest zdanie. " * 200  # ~3600 chars
        script = PodcastScript(
            title="Long Episode",
            chunks=[
                TTSChunk(
                    id="long_1",
                    text=long_text,
                    section_number=1,
                    section_title="Długi temat",
                ),
            ],
        )

        cfg = OpenAITTSConfig(
            api_key="sk-fake-key",
            output_dir="output/audio/test_long",
            max_chars_per_call=1000,
        )

        result = generate_podcast_audio(script, cfg)

        # Should still succeed (multiple API calls internally)
        assert len(result) == 1
        assert mock_client.audio.speech.create.call_count > 1


# ── Manifest loading tests ────────────────────────────────────


class TestLoadManifest:
    """Tests for manifest loading."""

    def test_loads_valid_manifest(self, tmp_path: Path) -> None:
        manifest_data = {
            "title": "Test",
            "metadata": {"total_sections": 1, "total_chunks": 1, "total_characters": 10},
            "chunks": [
                {
                    "id": "intro",
                    "text": "Hello",
                    "section_number": 0,
                    "section_title": "Intro",
                    "is_intro": True,
                    "is_outro": False,
                    "is_transition": False,
                    "audio_file": str(tmp_path / "intro.wav"),
                }
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        result = load_manifest(manifest_path)
        assert result["title"] == "Test"
        assert len(result["chunks"]) == 1

    def test_missing_manifest_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest(Path("/nonexistent/manifest.json"))
