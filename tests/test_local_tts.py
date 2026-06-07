"""Tests for the local TTS module."""

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
    LocalTTSConfig,
    generate_podcast_audio_local,
    generate_test_samples,
    load_manifest_local,
)

# ── Config tests ──────────────────────────────────────────────


class TestLocalTTSConfig:
    """Tests for LocalTTSConfig validation."""

    def test_default_config(self) -> None:
        cfg = LocalTTSConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8000
        assert cfg.language == "pl"
        assert cfg.voice == "default"
        assert cfg.emotion == "neutral"
        assert cfg.max_retries == 10
        assert cfg.retry_delay == 2.0

    def test_custom_config(self) -> None:
        cfg = LocalTTSConfig(host="192.168.1.10", port=9000, language="en")
        assert cfg.host == "192.168.1.10"
        assert cfg.port == 9000
        assert cfg.language == "en"

    def test_base_url_property(self) -> None:
        cfg = LocalTTSConfig(host="10.0.0.1", port=5000)
        assert cfg.base_url == "http://10.0.0.1:5000"

    def test_invalid_port_too_low(self) -> None:
        with pytest.raises(ValueError, match="Port must be between"):
            LocalTTSConfig(port=0)

    def test_invalid_port_too_high(self) -> None:
        with pytest.raises(ValueError, match="Port must be between"):
            LocalTTSConfig(port=65536)

    def test_invalid_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be at least"):
            LocalTTSConfig(max_retries=0)

    def test_invalid_retry_delay(self) -> None:
        with pytest.raises(ValueError, match="retry_delay must be positive"):
            LocalTTSConfig(retry_delay=-1.0)


# ── save_audio_file tests (local module) ──────────────────────


class TestLocalSaveAudioFile:
    """Tests for save_audio_file helper in local_tts."""

    def test_saves_wav_bytes(self, tmp_path: Path) -> None:
        from tts.local_tts import save_audio_file

        fake_wav = b"RIFF....WAV"
        result = save_audio_file(fake_wav, tmp_path / "test.wav")
        assert result.exists()
        assert result.read_bytes() == fake_wav

    def test_wraps_raw_data_in_wav(self, tmp_path: Path) -> None:
        from tts.local_tts import save_audio_file

        fake_data = b"raw audio bytes"
        result = save_audio_file(fake_data, tmp_path / "test.wav")
        assert result.exists()
        assert result.read_bytes() != fake_data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        from tts.local_tts import save_audio_file

        nested = tmp_path / "a" / "b" / "c.wav"
        result = save_audio_file(b"RIFF....WAV", nested)
        assert result.exists()


# ── Helpers ───────────────────────────────────────────────────


def _health_response() -> MagicMock:
    """Mock response for /health."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok"}
    return resp


def _audio_response(content: bytes = b"fake_audio") -> MagicMock:
    """Mock response for /jobs/audio/{job_id}."""
    resp = MagicMock()
    resp.status_code = 200
    resp.content = content
    return resp


def _status_response(job_id: str, status: str, error: str | None = None) -> MagicMock:
    """Mock response for /jobs/{job_id} (status check)."""
    resp = MagicMock()
    resp.status_code = 200
    data: dict[str, object] = {"job_id": job_id, "status": status}
    if error:
        data["error_message"] = error
    resp.json.return_value = data
    return resp


def _dispatch_get(url: str, job_id: str = "test-job", **kwargs: object) -> MagicMock:
    """Dispatch mocked GET requests to the correct response based on URL."""
    if "/health" in url:
        return _health_response()
    if "/audio/" in url:
        return _audio_response()
    return _status_response(job_id, "completed")


def _make_post_response(job_id: str = "test-job") -> MagicMock:
    """Mock response for POST /jobs/."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"job_id": job_id, "status": "pending"}
    return resp


# ── Integration tests (mocked API) ────────────────────────────


class TestGeneratePodcastAudioLocal:
    """End-to-end test with mocked local TTS server API."""

    @patch("tts.local_tts.requests")
    def test_generates_audio_for_chunks(self, mock_requests: MagicMock) -> None:
        """Successful job flow: health → create → status→ download."""
        mock_requests.get.side_effect = lambda url, **kw: _dispatch_get(url, "chunk-intro")
        mock_requests.post.return_value = _make_post_response("chunk-intro")

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

        cfg = LocalTTSConfig(
            host="127.0.0.1",
            port=8000,
            output_dir="output/audio/test_local_mock",
            max_retries=3,
        )

        result = generate_podcast_audio_local(script, cfg)

        assert len(result) == 2
        assert "intro" in result
        assert "chunk_1_0" in result

        manifest_path = Path(cfg.output_dir) / "manifest.json"
        assert manifest_path.exists()

    @patch("tts.local_tts.requests")
    def test_handles_failed_job(self, mock_requests: MagicMock) -> None:
        """Failed jobs are logged and skipped."""

        def _dispatch_get(url: str, job_id: str = "failed-job", **kwargs: object) -> MagicMock:
            if "/health" in url:
                return _health_response()
            if "/audio/" in url:
                return _audio_response()
            return _status_response(job_id, "failed", error="Model error")

        mock_requests.get.side_effect = _dispatch_get
        mock_requests.post.return_value = _make_post_response("failed-job")

        script = PodcastScript(
            title="Test Episode",
            chunks=[
                TTSChunk(
                    id="intro",
                    text="Witajcie.",
                    section_number=0,
                    section_title="Wstęp",
                    is_intro=True,
                ),
            ],
        )

        cfg = LocalTTSConfig(
            output_dir="output/audio/test_fail",
            max_retries=1,
        )

        result = generate_podcast_audio_local(script, cfg)

        assert len(result) == 0


# ── Manifest loading tests (local) ────────────────────────────


class TestLoadManifestLocal:
    """Tests for manifest loading from local TTS."""

    def test_loads_valid_manifest(self, tmp_path: Path) -> None:
        manifest_data = {
            "title": "Test",
            "metadata": {"total_sections": 1, "total_chunks": 1, "total_characters": 10},
            "chunks": [
                {
                    "id": "intro",
                    "text": "Witaj",
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

        result = load_manifest_local(manifest_path)
        assert result["title"] == "Test"
        assert len(result["chunks"]) == 1

    def test_missing_manifest_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest_local(Path("/nonexistent/manifest.json"))


# ── generate_test_samples tests (mocked) ──────────────────────


class TestGenerateTestSamples:
    """Tests for test sample generation with mocked server."""

    @patch("tts.local_tts.requests")
    def test_generates_sample_wavs(self, mock_requests: MagicMock) -> None:
        """generate_test_samples creates WAV files."""
        mock_requests.get.side_effect = lambda url, **kw: _dispatch_get(url, "test-job")
        mock_requests.post.return_value = _make_post_response("test-job")

        results = generate_test_samples(
            LocalTTSConfig(
                output_dir="tests/output/test_mock_samples",
                max_retries=3,
            ),
        )

        assert len(results) == 3
        for test_id, path in results.items():
            assert test_id.startswith("test_")
            assert path.endswith(".wav")
