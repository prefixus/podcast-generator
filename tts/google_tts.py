"""Text-to-speech generation using Google Cloud Text-to-Speech API.

Takes TTSChunk objects from the preprocess pipeline and generates
audio files via Google Cloud TTS. Audio files are saved as WAV.

Supports three authentication methods (tried in order):
  1. OAuth 2.0 client secret file (via oauth_client_secret_file param)
  2. API key (via GOOGLE_API_KEY env var, passed as query param)
  3. Service account JSON key file (via credentials_file parameter)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from google_auth_oauthlib.flow import InstalledAppFlow

from preprocess.tts_script_builder import PodcastScript, TTSChunk

# OAuth scopes for Google Cloud Text-to-Speech
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


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


def _get_api_key() -> str | None:
    """Get Google API key from environment variable."""
    return os.environ.get("GOOGLE_API_KEY")


def _get_oauth_client_secret_file() -> str | None:
    """Get OAuth client secret file path from environment variable or default."""
    env_path = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET_FILE")
    if env_path:
        return env_path
    # Default location in project root
    default = Path("google_oauth_client_secret.json")
    if default.exists():
        return str(default)
    return None


def _get_credentials_path() -> str | None:
    """Get credentials file path from environment variable."""
    return os.environ.get("GOOGLE_CREDENTIALS_FILE")


def _get_oauth_credentials() -> Any | None:
    """Get OAuth 2.0 credentials from client secret file.

    This prompts the user to authorize access in a browser.
    Credentials are cached in .tts_oauth_token.json for reuse.
    """
    client_secret_file = _get_oauth_client_secret_file()
    if not client_secret_file:
        return None

    # Check for cached token
    token_file = Path(".tts_oauth_token.json")
    if token_file.exists():
        from google.oauth2.credentials import Credentials
        return Credentials.from_authorized_user_file(str(token_file), SCOPES)

    # Load client config and run OAuth flow
    with open(client_secret_file, "r") as f:
        client_secrets = json.load(f)

    from google_auth_oauthlib.flow import Flow
    import webbrowser

    flow = Flow.from_client_config(
        client_secrets,
        scopes=SCOPES,
        redirect_uri="http://localhost",
    )

    authorization_url, _ = flow.authorization_url(prompt="consent")
    print("Please visit the following URL to authorize access:")
    print(authorization_url)
    print()
    print("After authorizing, paste the authorization code from the redirect URL.")
    print("(Look for the 'code=' parameter in the redirect URL)")
    print()

    # Read the authorization code from console
    authorization_response = input("Authorization code: ").strip()

    # Extract the code from a full redirect URL if needed
    if "code=" in authorization_response:
        from urllib.parse import urlparse
        parsed = urlparse(authorization_response)
        authorization_code = dict(__import__("urllib.parse").parse_qsl(parsed.query))["code"]
    else:
        authorization_code = authorization_response

    flow.fetch_token(code=authorization_code)

    # Cache the token as JSON for reuse
    _cache_credentials(flow.credentials, token_file)

    return flow.credentials


def _cache_credentials(credentials: Any, token_file: Path) -> None:
    """Cache OAuth credentials as JSON for reuse."""
    from google.oauth2.credentials import Credentials

    if isinstance(credentials, Credentials):
        token_json = credentials.to_json()
        token_file.write_text(token_json, encoding="utf-8")


def _make_authenticated_request(
    config: GoogleTTSConfig,
    text: str,
) -> bytes:
    """Make a Google Cloud TTS REST API call with OAuth 2.0 authentication."""
    credentials = _get_oauth_credentials()
    if not credentials:
        raise RuntimeError(
            "No authentication configured. Set GOOGLE_API_KEY for API key auth, "
            "or provide a Google OAuth client secret file (google_oauth_client_secret.json "
            "or via GOOGLE_OAUTH_CLIENT_SECRET_FILE env var) for OAuth 2.0 auth."
        )

    # Get an access token
    import google.oauth2.credentials
    if isinstance(credentials, google.oauth2.credentials.Credentials):
        import google.auth.transport.requests
        credentials.refresh(google.auth.transport.requests.Request())
        access_token = credentials.token
    else:
        # Assume it's a valid credentials object with token
        access_token = getattr(credentials, "token", None)
        if not access_token:
            raise RuntimeError("Could not obtain access token from credentials")

    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": config.language_code,
            "name": config.voice_name,
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16" if config.audio_encoding == "LINEAR16" else "MP3",
            "speakingRate": config.speaking_rate,
            "pitch": config.pitch,
        },
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    result = response.json()
    audio_content_b64 = result["audioContent"]
    return _b64decode(audio_content_b64)


def _synthesize_via_rest(
    config: GoogleTTSConfig,
    text: str,
) -> bytes:
    """Make a Google Cloud TTS REST API call and return audio bytes.

    Tries authentication methods in order:
    1. OAuth 2.0 (client secret file)
    2. API key (GOOGLE_API_KEY env var)
    3. Service account credentials (credentials_file parameter)
    """
    # Try OAuth 2.0 first
    oauth_file = _get_oauth_client_secret_file()
    if oauth_file:
        try:
            return _make_authenticated_request(config, text)
        except Exception as e:
            print(f"OAuth auth failed: {e}, trying API key...")

    # Try API key
    api_key = _get_api_key()
    if api_key:
        url = (
            f"https://texttospeech.googleapis.com/v1/text:synthesize"
            f"?key={api_key}"
        )
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": config.language_code,
                "name": config.voice_name,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16" if config.audio_encoding == "LINEAR16" else "MP3",
                "speakingRate": config.speaking_rate,
                "pitch": config.pitch,
            },
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        audio_content_b64 = result["audioContent"]
        return _b64decode(audio_content_b64)

    # Fallback: use gRPC client with service account credentials
    from google.cloud import texttospeech as _tts_module

    credentials_path = _get_credentials_path() or config.credentials_file
    kwargs: dict[str, Any] = {}
    if credentials_path:
        kwargs["credentials_file"] = credentials_path
    client = _tts_module.TextToSpeechClient(**kwargs)

    input_text = _tts_module.SynthesisInput(text=text)
    voice_params = _tts_module.VoiceSelectionParams(
        language_code=config.language_code,
        name=config.voice_name,
    )
    audio_config = _tts_module.AudioConfig(
        audio_encoding=_tts_module.AudioEncoding.LINEAR16,
        speaking_rate=config.speaking_rate,
        pitch=config.pitch,
    )
    request = _tts_module.SynthesizeSpeechRequest(
        input=input_text,
        voice=voice_params,
        audio_config=audio_config,
    )
    response = client.synthesize_speech(request=request)
    return response.audio_content


def _b64decode(b64: str) -> bytes:
    """Base64 decode a string."""
    import base64
    return base64.b64decode(b64)


def generate_audio_chunk(
    chunk: TTSChunk,
    config: GoogleTTSConfig,
) -> bytes:
    """Send a single TTSChunk text to Google Cloud TTS and return audio bytes.

    If the text exceeds max_chars_per_call, the text is split into
    smaller segments, each sent separately, and the results are
    concatenated.
    """
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
                all_audio.append(_synthesize_via_rest(config, " ".join(current_segment)))
                current_segment = [seg]
                current_len = len(seg)
            else:
                current_segment.append(seg)
                current_len += len(seg)

        if current_segment:
            all_audio.append(_synthesize_via_rest(config, " ".join(current_segment)))

        return b"".join(all_audio)

    return _synthesize_via_rest(config, text)


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
