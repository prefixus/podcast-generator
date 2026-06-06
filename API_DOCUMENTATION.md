# TTS API Documentation

Local FastAPI server for Higgs Audio v3 TTS 4B with concurrent request processing and job queuing.

## Quick Start

### Running the Server

```bash
# Install dependencies (if not already done)
uv sync

# Start the server
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The server will automatically start a background worker to process TTS generation jobs.

## API Endpoints

### Health Check
```http
GET /health
```

Returns server health status.

**Example:**
```bash
curl http://127.0.0.1:8000/health
# Output: {"status": "ok"}
```

### Create TTS Generation Job (Async)

```http
POST /jobs/
Content-Type: application/json
```

Creates a new TTS generation job and adds it to the queue. Returns immediately with job ID.

**Request Body:**
```json
{
  "text": "Cześć, jak się masz?",
  "language": "pl",
  "voice": "default",
  "emotion": "neutral"
}
```

**Parameters:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| text | string | Yes | - | Text to synthesize (Polish recommended for best quality) |
| language | string | No | "pl" | Language code (e.g., "pl", "en") |
| voice | string | No | "default" | Voice profile identifier |
| emotion | string | No | "neutral" | Emotion for synthesis |

**Response (200 OK):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "text": "Cześć, jak się masz?",
  "language": "pl",
  "voice": "default",
  "emotion": "neutral",
  "status": "pending"
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/jobs/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Witaj świecie!", "language": "pl"}' | jq
```

### Get Job Status

```http
GET /jobs/{job_id}
```

Retrieves the status and details of a specific job.

**Response (200 OK):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "text": "Witaj świecie!",
  "language": "pl",
  "voice": "default",
  "emotion": "neutral",
  "status": "completed",
  "result_path": "/path/to/output/tts_a1b2c3d4.wav",
  "error_message": null
}
```

**Response (404 Not Found):**
```json
{
  "detail": "Job {job_id} not found"
}
```

**Example:**
```bash
curl http://127.0.0.1:8000/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Download Generated Audio

```http
GET /jobs/audio/{job_id}
```

Downloads the generated audio file for a completed job.

**Response (200 OK):**
- Content-Type: `audio/wav`
- Header: `Content-Disposition: attachment; filename="tts_{job_id}.wav"`

**Response (400 Bad Request):**
```json
{
  "detail": "Job is not completed. Current status: processing"
}
```

**Response (404 Not Found):**
```json
{
  "detail": "Job {job_id} not found"
}
```

**Example:**
```bash
curl -o output.wav http://127.0.0.1:8000/jobs/audio/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Get All Jobs

```http
GET /jobs/
```

Retrieves all jobs in the queue (oldest first).

**Example:**
```bash
curl http://127.0.0.1:8000/jobs/
```

### Get Recent Jobs

```http
GET /jobs/recent?limit=10
```

Retrieves the most recently completed or failed jobs.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 10 | Maximum number of jobs to return |

**Example:**
```bash
curl http://127.0.0.1:8000/jobs/recent?limit=5
```

### Get Queue Status

```http
GET /jobs/status
```

Retrieves current queue statistics.

**Response (200 OK):**
```json
{
  "total_jobs": 15,
  "pending": 2,
  "processing": 1,
  "completed": 11,
  "failed": 1,
  "queue_length": 3
}
```

**Example:**
```bash
curl http://127.0.0.1:8000/jobs/status
```

### Worker Status

```http
GET /jobs/worker/status
```

Retrieves worker status and queue length.

**Response (200 OK):**
```json
{
  "running": true,
  "job_queue_length": 3
}
```

**Example:**
```bash
curl http://127.0.0.1:8000/jobs/worker/status
```

### Control Worker

```http
POST /jobs/worker/start
```
Starts the worker (if stopped).

```http
POST /jobs/worker/stop
```
Stops the worker.

**Response (200 OK):**
```json
{
  "status": "started",
  "message": "Worker started successfully"
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/jobs/worker/start
```

## Job Status Lifecycle

```
pending → processing → completed (with result_path)
         ↓
       failed (with error_message)
```

### Status Values
- `pending`: Job is queued but not yet being processed
- `processing`: Job is currently being synthesized
- `completed`: Job finished successfully (audio file available)
- `failed`: Job failed with error message

## Multiple Request Handling

The API supports multiple concurrent requests through:

1. **Job Queue**: Requests are queued and processed in order
2. **Background Worker**: Asynchronous processing of jobs
3. **Non-blocking Responses**: Jobs return immediately with status

### Example: Batch Processing

```bash
# Create multiple jobs
JOB1=$(curl -s -X POST http://127.0.0.1:8000/jobs/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Zdanie pierwsze", "language": "pl"}' | jq -r '.job_id')

JOB2=$(curl -s -X POST http://127.0.0.1:8000/jobs/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Zdanie drugie", "language": "pl"}' | jq -r '.job_id')

JOB3=$(curl -s -X POST http://127.0.0.1:8000/jobs/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Zdanie trzecie", "language": "pl"}' | jq -r '.job_id')

echo "Created jobs: $JOB1, $JOB2, $JOB3"

# Check queue status
curl http://127.0.0.1:8000/jobs/status

# Wait a bit for processing, then download
sleep 5
curl -o chapter1.wav http://127.0.0.1:8000/jobs/audio/$JOB1
curl -o chapter2.wav http://127.0.0.1:8000/jobs/audio/$JOB2
curl -o chapter3.wav http://127.0.0.1:8000/jobs/audio/$JOB3
```

## Python Client Example

```python
import requests
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def create_job(text: str, language: str = "pl") -> dict:
    """Create a new TTS generation job."""
    response = requests.post(
        f"{BASE_URL}/jobs/",
        json={"text": text, "language": language},
    )
    response.raise_for_status()
    return response.json()

def get_job_status(job_id: str) -> dict:
    """Get job status."""
    response = requests.get(f"{BASE_URL}/jobs/{job_id}")
    response.raise_for_status()
    return response.json()

def download_audio(job_id: str, output_path: Path) -> bytes:
    """Download generated audio."""
    response = requests.get(f"{BASE_URL}/jobs/audio/{job_id}")
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    return response.content

# Example usage
if __name__ == "__main__":
    # Create jobs for multiple chapters
    chapter_texts = [
        "To jest rozdzia³ pierwszy.",
        "To jest rozdzia³ drugi.",
        "To jest rozdzia³ trzeci.",
    ]
    
    job_ids = []
    for text in chapter_texts:
        response = create_job(text)
        job_ids.append(response["job_id"])
        print(f"Created job: {response['job_id']}")
    
    # Wait for processing
    print("Waiting for jobs to complete...")
    while True:
        status = requests.get(f"{BASE_URL}/jobs/status").json()
        print(f"Queue status: {status}")
        
        if status["pending"] == 0 and status["processing"] == 0:
            break
        time.sleep(2)
    
    # Download completed audio files
    for i, job_id in enumerate(job_ids):
        status = get_job_status(job_id)
        if status["status"] == "completed":
            output_path = Path(f"chapter_{i+1}.wav")
            download_audio(job_id, output_path)
            print(f"Downloaded: {output_path}")
        else:
            print(f"Job {job_id} failed: {status.get('error_message', 'Unknown error')}")
```

## Configuration

Edit `.env` file to configure the server:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `multimodalart/higgs-audio-v3-tts-4b-transformers` | HuggingFace model repository |
| `DEVICE` | `cpu` | Device: `mps` (Apple GPU) or `cpu` |
| `TTS_HOST` | `127.0.0.1` | Bind address |
| `TTS_PORT` | `8000` | Bind port |
| `REFERENCE_AUDIO_PATH` | `app/inference/reference.wav` | Path to reference audio for voice cloning |
| `REFERENCE_TEXT` | (see config.py) | Reference text for voice cloning |

## Notes

- Jobs are processed sequentially by a background worker
- Completed audio files are saved to the `output/` directory
- Failed jobs include error messages in their status
- Worker starts automatically when server starts
- All requests are async and return immediately
