"""Download all completed jobs from the Higgs TTS server.

This script queries the server for all jobs, identifies completed ones,
and downloads their audio files to the output directory.
"""

import wave
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = Path("output/audio")


def get_all_jobs() -> list[dict]:
    """Fetch all jobs from the server."""
    resp = requests.get(f"{BASE_URL}/jobs/", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_recent_jobs(limit: int = 100) -> list[dict]:
    """Fetch recently completed/failed jobs."""
    resp = requests.get(f"{BASE_URL}/jobs/recent?limit={limit}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def download_audio(job_id: str) -> bytes | None:
    """Download audio for a completed job."""
    resp = requests.get(f"{BASE_URL}/jobs/audio/{job_id}", timeout=60)
    if resp.status_code == 400:
        status_resp = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=10)
        status_resp.raise_for_status()
        status = status_resp.json()
        print(f"  Job {job_id[:8]}... is not completed: {status.get('status')}")
        return None
    if resp.status_code == 404:
        print(f"  Job {job_id[:8]}... not found")
        return None
    resp.raise_for_status()
    return resp.content


def save_wav(audio_bytes: bytes, output_path: Path) -> Path:
    """Wrap raw Opus audio in a WAV container."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_bytes)
    output_path.write_bytes(wav_buffer.getvalue())
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all jobs from server
    print("Fetching all jobs from server...")
    all_jobs = get_all_jobs()
    print(f"Total jobs on server: {len(all_jobs)}")

    # Categorize jobs by status
    completed_jobs = []
    failed_jobs = []
    processing_jobs = []
    pending_jobs = []

    for job in all_jobs:
        status = job.get("status", "unknown")
        job_id = job.get("job_id", "unknown")
        text = job.get("text", "")[:80]

        if status == "completed":
            completed_jobs.append(job)
        elif status == "failed":
            failed_jobs.append(job)
        elif status == "processing":
            processing_jobs.append(job)
        elif status == "pending":
            pending_jobs.append(job)
        else:
            print(f"  Unknown status '{status}' for job {job_id[:8]}...")

    print("\n=== Job Summary ===")
    print(f"  Completed:  {len(completed_jobs)}")
    print(f"  Failed:     {len(failed_jobs)}")
    print(f"  Processing: {len(processing_jobs)}")
    print(f"  Pending:    {len(pending_jobs)}")

    # Download all completed jobs
    print(f"\n=== Downloading {len(completed_jobs)} completed jobs ===")
    downloaded = 0
    download_errors = 0

    for job in completed_jobs:
        job_id = job.get("job_id", "")
        text = job.get("text", "")[:60]
        print(f"  Downloading: {job_id[:8]}... ({text}...)")

        audio_bytes = download_audio(job_id)
        if audio_bytes:
            output_path = OUTPUT_DIR / f"{job_id}.wav"
            try:
                save_wav(audio_bytes, output_path)
                downloaded += 1
                print(f"    -> Saved: {output_path.name} ({output_path.stat().st_size / 1024:.0f} KB)")
            except Exception as e:
                print(f"    -> SAVE ERROR: {e}")
                download_errors += 1
        else:
            print("    -> DOWNLOAD FAILED")
            download_errors += 1

    # Print failed jobs summary
    if failed_jobs:
        print(f"\n=== {len(failed_jobs)} FAILED Jobs ===")
        for job in failed_jobs:
            job_id = job.get("job_id", "unknown")
            text = job.get("text", "")[:80]
            error_msg = job.get("error_message", "No error message")
            print(f"  Job: {job_id[:8]}...")
            print(f"    Text: {text}")
            print(f"    Error: {error_msg}")
            print()

    # Print processing/pending jobs
    if processing_jobs:
        print(f"\n=== {len(processing_jobs)} Processing Jobs ===")
        for job in processing_jobs:
            job_id = job.get("job_id", "unknown")
            text = job.get("text", "")[:80]
            print(f"  Job: {job_id[:8]}... - {text}")

    if pending_jobs:
        print(f"\n=== {len(pending_jobs)} Pending Jobs ===")
        for job in pending_jobs:
            job_id = job.get("job_id", "unknown")
            text = job.get("text", "")[:80]
            print(f"  Job: {job_id[:8]}... - {text}")

    # Print final stats
    print("\n=== Final Stats ===")
    print(f"  Downloaded:    {downloaded}")
    print(f"  Download errs: {download_errors}")
    print(f"  Failed jobs:   {len(failed_jobs)}")
    print(f"  Output dir:    {OUTPUT_DIR}")

    existing = list(OUTPUT_DIR.glob("*.wav"))
    print(f"  Total .wav files: {len(existing)}")
    total_size = sum(f.stat().st_size for f in existing)
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    import io  # needed for save_wav

    main()
