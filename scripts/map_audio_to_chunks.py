"""Map downloaded audio files (by job_id) to chunk IDs from the podcast script.

Reads the script JSON, matches job_id prefixes to chunk IDs,
and updates the manifest with correct .wav filenames.
"""

import json
from pathlib import Path

SCRIPT_JSON = Path("output/Seksuologia_opracowane_tezy_script.json")
AUDIO_DIR = Path("output/audio")
MANIFEST = AUDIO_DIR / "manifest.json"


def main():
    # Load the script JSON to get chunk IDs
    script = json.loads(SCRIPT_JSON.read_text(encoding="utf-8"))
    chunks = script["chunks"]

    # Get all downloaded .wav files from server (by job_id)
    wav_files = [f for f in AUDIO_DIR.glob("*.wav") if len(f.stem) == 36]  # UUID format
    print(f"Found {len(wav_files)} server-generated .wav files (by job_id)")

    # Load existing manifest (from the running pipeline)
    existing_manifest = None
    if MANIFEST.exists():
        existing_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # Map job_id -> chunk_id by matching text content
    chunk_to_job_id: dict[str, str] = {}
    for chunk in chunks:
        chunk_text = chunk["text"]
        for wav_file in wav_files:
            job_id = wav_file.stem
            # Get job text from server to match
            import requests
            try:
                resp = requests.get(f"http://127.0.0.1:8000/jobs/{job_id}", timeout=10)
                job = resp.json()
                if job.get("text", "")[:30] == chunk_text[:30]:
                    chunk_to_job_id[chunk["id"]] = job_id
                    break
            except Exception:
                pass

    print(f"\nMapping {len(chunk_to_job_id)} chunks to job_ids")

    # Update manifest with correct wav filenames
    if existing_manifest:
        for chunk_data in existing_manifest["chunks"]:
            chunk_id = chunk_data["id"]
            if chunk_id in chunk_to_job_id:
                job_id = chunk_to_job_id[chunk_id]
                wav_path = AUDIO_DIR / f"{job_id}.wav"
                if wav_path.exists():
                    chunk_data["audio_file"] = str(wav_path)
                    chunk_data["status"] = "ok"
                    chunk_data["job_id"] = job_id
                    size_kb = wav_path.stat().st_size / 1024
                    print(f"  {chunk_id} -> {job_id[:8]}... ({size_kb:.0f} KB)")

    MANIFEST.write_text(
        json.dumps(existing_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nManifest updated: {MANIFEST}")

    # Count results
    total = len(existing_manifest["chunks"])
    with_audio = sum(1 for c in existing_manifest["chunks"] if c["audio_file"])
    failed = sum(1 for c in existing_manifest["chunks"] if c.get("status") == "failed")
    print(f"Total chunks: {total}")
    print(f"With audio:   {with_audio}")
    print(f"Failed:       {failed}")


if __name__ == "__main__":
    import requests
    main()
