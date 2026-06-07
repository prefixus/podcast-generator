"""Tests for the Google Cloud TTS module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.tts_script_builder import PodcastScript, TTSChunk
from tts import (
    GoogleTTSConfig,
    generate_podcast_audio_google,
    load_manifest_google,
)

# ── Config tests ──────────────────────────────────────────────


class TestGoogleTTSConfig:
    """Tests for GoogleTTSConfig validation."""

    def test_default_config(self) -> None:
        cfg = GoogleTTSConfig()
        assert cfg.language_code == "pl-PL"
        assert cfg.voice_name == "pl-PL-Wavenet-A"
        assert cfg.speaking_rate == 1.0
        assert cfg.pitch == 0.0
        assert cfg.audio_encoding == "LINEAR16"
        assert cfg.max_chars_per_call == 4096

    def test_invalid_speaking_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="speaking_rate must be between"):
            GoogleTTSConfig(speaking_rate=5.0)

    def test_invalid_pitch_raises(self) -> None:
        with pytest.raises(ValueError, match="pitch must be between"):
            GoogleTTSConfig(pitch=25.0)

    def test_custom_output_dir(self) -> None:
        cfg = GoogleTTSConfig(output_dir="custom/audio/path")
        assert cfg.output_dir == "custom/audio/path"


# ── save_audio_file tests (google module) ─────────────────────


class TestSaveAudioFileGoogle:
    """Tests for save_audio_file helper in google_tts."""

    def test_saves_wav_bytes(self, tmp_path: Path) -> None:
        from tts.google_tts import save_audio_file

        fake_wav = b"RIFF....WAV"
        result = save_audio_file(fake_wav, tmp_path / "test.wav")
        assert result.exists()
        assert result.read_bytes() == fake_wav

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        from tts.google_tts import save_audio_file

        nested = tmp_path / "a" / "b" / "c.wav"
        result = save_audio_file(b"RIFF....WAV", nested)
        assert result.exists()


# ── Integration tests (mocked API) ────────────────────────────


class TestGeneratePodcastAudioGoogle:
    """End-to-end test with mocked Google Cloud TTS API."""

    @patch("tts.google_tts.texttospeech.TextToSpeechClient")
    def test_generates_audio_for_chunks(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the synthesize_speech response
        mock_response = MagicMock()
        mock_response.audio_content = b"fake_google_audio"
        mock_client.synthesize_speech.return_value = mock_response

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

        cfg = GoogleTTSConfig(
            output_dir="output/audio/test_google_mock",
        )

        result = generate_podcast_audio_google(script, cfg)

        assert len(result) == 2
        assert "intro" in result
        assert "chunk_1_0" in result

        # Check manifest was saved
        manifest_path = Path(cfg.output_dir) / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["title"] == "Test Episode"
        assert len(manifest["chunks"]) == 2

    @patch("tts.google_tts.texttospeech.TextToSpeechClient")
    def test_handles_oversized_chunk(self, mock_client_class: MagicMock) -> None:
        """Oversized chunks are split and multiple API calls are made."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.audio_content = b"fake_google_audio"
        mock_client.synthesize_speech.return_value = mock_response

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

        cfg = GoogleTTSConfig(
            output_dir="output/audio/test_google_long",
            max_chars_per_call=1000,
        )

        result = generate_podcast_audio_google(script, cfg)

        # Should still succeed (multiple API calls internally)
        assert len(result) == 1
        assert mock_client.synthesize_speech.call_count > 1


# ── Manifest loading tests (google) ───────────────────────────


class TestLoadManifestGoogle:
    """Tests for manifest loading from Google TTS."""

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

        result = load_manifest_google(manifest_path)
        assert result["title"] == "Test"
        assert len(result["chunks"]) == 1

    def test_missing_manifest_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest_google(Path("/nonexistent/manifest.json"))
