"""Basic smoke tests for podcast-generator."""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_main_runs() -> None:
    """Verify main() executes without error."""
    from main import main

    main()  # expects no exceptions
