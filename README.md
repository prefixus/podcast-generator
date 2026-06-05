# podcast-generator

Generate speech-to-text audio files from documents using the OpenAI API.

## Goal

Build a simple application that:

1. **File selection** — pick input files (PDF, text, etc.) via a file selector
2. **Text extraction** — process and prepare the file content for text-to-speech
3. **Text-to-speech** — send extracted text to an API (OpenAI first) and receive audio output
4. **Iterative improvement** — refine the pipeline over time

## Current State

Phase 1 — **Project scaffolding & API integration**. The foundation is in place:

- OpenAI SDK installed and ready for text-to-speech API calls
- Development tooling configured (ruff, mypy, pytest, poe)
- Basic test infrastructure exists

## Setup

Requires **Python 3.11+** and [uv](https://github.com/astral-sh/uv).

```bash
# Install dependencies
uv sync

# Update lockfile after any dependency changes
uv lock
```

## Development

All tasks are managed via [Poe the Poet](https://github.com/mhammond/poethepoet).

| Command | Description |
|---------|-------------|
| `uv run poe dev` | Run everything — format, test, lint |
| `uv run poe dev-format` | Run ruff linter and formatter |
| `uv run poe dev-test` | Run pytest |
| `uv run poe dev-lint` | Run mypy type checks |

### Individual tools

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy .

# Test
uv run pytest
```

## Project Structure

```
.
├── main.py            # Entry point
├── pyproject.toml     # Project config, deps, tool settings, Poe tasks
├── uv.lock            # Locked dependency graph
└── tests/
    └── test_main.py   # Smoke test
```

## Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.11+ |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| API client | [OpenAI SDK](https://github.com/openai/openai-python) |
| Linter / formatter | [ruff](https://github.com/astral-sh/ruff) |
| Type checker | [mypy](https://mypy.readthedocs.io/) |
| Testing | [pytest](https://docs.pytest.org/) |
| Task runner | [poethepoet](https://github.com/mhammond/poethepoet) |
