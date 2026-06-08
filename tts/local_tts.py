"""Text-to-speech generation using local FastAPI server.

Connects to a locally running TTS server (e.g., Higgs Audio v3)
and submits TTS chunks for asynchronous audio generation.
"""

from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from preprocess.tts_script_builder import PodcastScript, TTSChunk


@dataclass
class LocalTTSConfig:
    """Configuration for local FastAPI TTS server."""

    host: str = "127.0.0.1"
    port: int = 8000
    language: str = "pl"
    voice: str = "default"
    emotion: str = "neutral"
    output_dir: str | Path = "output/audio"
    max_retries: int = 60
    retry_delay: float = 5.0

    @property
    def base_url(self) -> str:
        """Build the base URL for the TTS server."""
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        """Health check endpoint."""
        return f"{self.base_url}/health"

    @property
    def jobs_url(self) -> str:
        """Jobs endpoint."""
        return f"{self.base_url}/jobs"

    @property
    def status_url(self) -> str:
        """Queue status endpoint."""
        return f"{self.base_url}/jobs/status"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 1 <= self.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.retry_delay <= 0:
            raise ValueError("retry_delay must be positive")


def check_health(config: LocalTTSConfig) -> dict[str, Any]:
    """Check if the TTS server is running."""
    try:
        response = requests.get(config.health_url, timeout=10)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except requests.RequestException as e:
        raise RuntimeError(f"TTS server health check failed: {e}")


def create_job(config: LocalTTSConfig, text: str) -> dict[str, Any]:
    """Create a new TTS generation job."""
    payload = {
        "text": text,
        "language": config.language,
        "voice": config.voice,
        "emotion": config.emotion,
    }
    try:
        response = requests.post(
            config.jobs_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to create TTS job: {e}")


def get_job_status(config: LocalTTSConfig, job_id: str) -> dict[str, Any]:
    """Get status of a TTS generation job."""
    url = f"{config.jobs_url}/{job_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            raise ValueError(f"Job {job_id} not found")
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to get job status: {e}")


def download_audio(config: LocalTTSConfig, job_id: str) -> bytes:
    """Download generated audio for a completed job."""
    url = f"{config.jobs_url}/audio/{job_id}"
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 400:
            status = get_job_status(config, job_id)
            raise ValueError(f"Job is not completed. Current status: {status.get('status')}")
        if response.status_code == 404:
            raise ValueError(f"Job {job_id} not found")
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download audio: {e}")


def wait_for_job_completion(config: LocalTTSConfig, job_id: str) -> dict[str, Any]:
    """Wait for a job to complete with polling."""
    attempts = 0
    while attempts < config.max_retries:
        status = get_job_status(config, job_id)
        job_status = status.get("status")

        if job_status == "completed":
            return status

        if job_status == "failed":
            error_msg = status.get("error_message", "Unknown error")
            raise RuntimeError(f"Job {job_id} failed: {error_msg}")

        attempts += 1
        if attempts < config.max_retries:
            time.sleep(config.retry_delay)

    raise TimeoutError(f"Job {job_id} did not complete within timeout")


def generate_audio_chunk(config: LocalTTSConfig, chunk: TTSChunk) -> bytes:
    """Generate audio for a single TTS chunk using local server.

    BLOCKING version: creates job, waits for completion, downloads audio.
    Use generate_podcast_audio for batch async processing.
    """
    job_response = create_job(config, chunk.text)
    job_id = job_response.get("job_id")
    if not job_id:
        raise ValueError("No job_id returned from server")

    wait_for_job_completion(config, job_id)
    audio_bytes = download_audio(config, job_id)

    return audio_bytes


def generate_podcast_audio(
    script: PodcastScript,
    config: LocalTTSConfig | None = None,
) -> dict[str, str]:
    """Generate audio for every chunk in a PodcastScript using local TTS.

    Uses a batched approach optimized for single-worker Higgs servers:
    submits a small batch of jobs, waits for them to complete, then
    submits the next batch. This avoids overwhelming the server's
    single worker and prevents request timeouts.

    Returns a mapping of chunk id → saved file path.
    """
    if config is None:
        config = LocalTTSConfig()

    # Check server health
    try:
        health = check_health(config)
        print(f"TTS Server health: {health}")
    except RuntimeError as e:
        print(f"Warning: Could not check server health: {e}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Batch size: match the server's single-worker capacity.
    # Submit 2 jobs at a time, wait for both to complete, then next batch.
    batch_size = 2

    chunks = script.chunks
    total = len(chunks)
    results: dict[str, str] = {}
    failed_chunks: list[str] = []
    processed = 0

    print(f"Processing {total} chunks in batches of {batch_size}...")

    while processed < total:
        # Submit a batch
        batch = chunks[processed : processed + batch_size]
        batch_chunk_to_job: dict[str, str] = {}  # chunk_id -> job_id (local to this batch)

        for chunk in batch:
            print(f"  Submitting: {chunk.id} ({len(chunk.text)} chars)")
            try:
                job_response = create_job(config, chunk.text)
                job_id = job_response.get("job_id")
                if job_id:
                    batch_chunk_to_job[chunk.id] = job_id
                else:
                    failed_chunks.append(chunk.id)
                    print(f"    ERROR: No job_id returned for {chunk.id}")
            except Exception as e:
                failed_chunks.append(chunk.id)
                print(f"    ERROR: {e}")

        if not batch_chunk_to_job:
            processed += len(batch)
            continue

        # Wait for the entire batch to complete
        print(f"  Waiting for {len(batch_chunk_to_job)} batch job(s) to complete...")
        batch_completed = 0

        while batch_completed < len(batch_chunk_to_job):
            newly_done: list[str] = []

            for chunk_id, job_id in batch_chunk_to_job.items():
                if chunk_id in results:
                    continue  # already downloaded

                try:
                    status = get_job_status(config, job_id)
                    job_status = status.get("status")

                    if job_status == "completed":
                        newly_done.append(chunk_id)
                    elif job_status == "failed":
                        error_msg = status.get("error_message", "Unknown error")
                        print(f"  [FAIL] {chunk_id}: {error_msg}")
                        if chunk_id not in failed_chunks:
                            failed_chunks.append(chunk_id)

                except Exception as e:
                    print(f"  [ERROR checking {chunk_id}]: {e}")

            # Download completed audio
            for chunk_id in newly_done:
                try:
                    job_id = batch_chunk_to_job[chunk_id]
                    audio_bytes = download_audio(config, job_id)
                    audio_path = save_audio_file(audio_bytes, output_dir / f"{chunk_id}.wav")
                    results[chunk_id] = str(audio_path)
                    batch_completed += 1
                    print(f"  [DONE] {chunk_id} -> {audio_path.name}")
                except Exception as e:
                    print(f"  [DOWNLOAD FAIL] {chunk_id}: {e}")
                    if chunk_id not in failed_chunks:
                        failed_chunks.append(chunk_id)

            if batch_completed < len(batch_chunk_to_job):
                time.sleep(config.retry_delay)

        processed += len(batch)
        print(f"  Progress: {processed}/{total} submitted, {len(results)} completed")

    _save_manifest(script, output_dir, results, failed_chunks)
    print(f"\nCompleted: {len(results)}/{total} chunks generated successfully.")
    return results


def _save_manifest(
    script: PodcastScript,
    output_dir: Path,
    results: dict[str, str],
    failed_chunks: list[str] | None = None,
) -> None:
    """Save the manifest JSON with audio file mappings."""
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
                "audio_file": results.get(chunk.id, ""),
                "status": "failed" if chunk.id in (failed_chunks or []) else "ok",
            }
            for chunk in script.chunks
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest saved to: {manifest_path}")


def _save_empty_manifest(script: PodcastScript, output_dir: Path) -> None:
    """Save an empty manifest when no jobs were submitted."""
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
                "audio_file": "",
                "status": "failed",
            }
            for chunk in script.chunks
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest saved to: {manifest_path}")


def save_audio_file(audio_bytes: bytes, output_path: str | Path) -> Path:
    """Save audio bytes to disk, creating directories as needed.

    Wraps raw Opus audio in a WAV container if needed.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Check if already WAV
    if output.suffix.lower() == ".wav":
        if audio_bytes[:4] == b"RIFF":
            output.write_bytes(audio_bytes)
            return output

    # Wrap in WAV container
    wav_data = _generate_wav_from_bytes(audio_bytes)
    output.write_bytes(wav_data)
    return output


def _generate_wav_from_bytes(raw_audio: bytes) -> bytes:
    """Wrap raw Opus audio bytes into a minimal WAV container."""
    import io

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(raw_audio)
    return wav_buffer.getvalue()


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a previously generated manifest for chapter merging."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def generate_single_test_chunk(config: LocalTTSConfig, text: str) -> tuple[str, bytes]:
    """Generate audio for a single test chunk and return (job_id, audio_bytes)."""
    job_response = create_job(config, text)
    job_id = job_response.get("job_id")
    if not job_id:
        raise ValueError("No job_id returned from server")

    wait_for_job_completion(config, job_id)
    audio_bytes = download_audio(config, job_id)

    return job_id, audio_bytes


def generate_test_samples(
    config: LocalTTSConfig | None = None,
    output_dir: str | Path = "tests/output",
) -> dict[str, str]:
    """Generate sample audio files for testing.

    Creates a few test chunks with Polish text and saves WAV files.
    """
    if config is None:
        config = LocalTTSConfig()

    test_texts = [
        ("test_1", "Witaj świecie! To jest przykładowy fragment do testowania."),
        ("test_2", "Generowanie mowy za pomocą lokalnego serwera TTS jest bardzo wygodne."),
        ("test_3", "Można generować wiele fragmentów jednocześnie i pobierać ich audio."),
    ]

    results: dict[str, str] = {}

    for test_id, text in test_texts:
        print(f"Generating: {test_id}")
        try:
            job_id, audio_bytes = generate_single_test_chunk(config, text)
            output_path = Path(output_dir) / f"{test_id}.wav"
            save_audio_file(audio_bytes, output_path)
            results[test_id] = str(output_path)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error with {test_id}: {e}")

    return results
