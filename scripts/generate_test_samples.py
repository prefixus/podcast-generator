"""Generate test audio samples from local TTS server.

Usage:
    python -m scripts.generate_test_samples
    python -m scripts.generate_test_samples --host 127.0.0.1 --port 8000 --output tests/output

This script connects to a locally running TTS server (e.g., Higgs Audio v3)
and generates a few test WAV files to validate the integration.
"""

from __future__ import annotations

import sys

from tts import LocalTTSConfig, generate_test_samples


def main() -> None:
    """Generate test audio samples from local TTS server."""
    args = list(sys.argv[1:])

    host = "127.0.0.1"
    port = 8000
    language = "pl"
    output_dir = "tests/output"

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 1
        elif arg == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"Warning: Invalid port '{args[i + 1]}', using default {port}")
            i += 1
        elif arg == "--language" and i + 1 < len(args):
            language = args[i + 1]
            i += 1
        elif arg == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 1
        i += 1

    config = LocalTTSConfig(
        host=host,
        port=port,
        language=language,
        output_dir=output_dir,
    )

    print(f"Connecting to TTS server: {config.base_url}")
    print(f"Language: {config.language}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        results = generate_test_samples(config)
        print()
        print("Generated files:")
        for test_id, path in results.items():
            print(f"  {test_id}: {path}")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure the TTS server is running at the specified host:port.")
        sys.exit(1)


if __name__ == "__main__":
    main()
