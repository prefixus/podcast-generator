"""Pytest configuration: per-test timing and skip-if-unrelated markers."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

# Map test module paths (relative to project root) to the source files they depend on.
# Slow tests are skipped unless their test file or source files are uncommitted.
_SLOW_TEST_DEPS: dict[str, list[str]] = {
    "tests/test_preprocess.py": [
        "preprocess/pdf_extractor.py",
        "preprocess/tts_script_builder.py",
    ],
    "tests/test_tts.py": [
        "tts/openai_tts.py",
    ],
    "tests/test_local_tts.py": [
        "tts/local_tts.py",
    ],
    "tests/test_google_tts.py": [
        "tts/google_tts.py",
    ],
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_uncommitted_files() -> set[str]:
    """Return set of relative file paths that are uncommitted (modified or untracked)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-z"],
            capture_output=True,
            text=False,
            check=False,
        )
        if result.returncode != 0:
            return set()
        output = result.stdout.decode("utf-8", errors="replace")
        uncommitted: set[str] = set()
        for line in output.split("\0"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                path = parts[1].strip()
                if " -> " in path:
                    path = path.split(" -> ", 1)[1].strip()
                uncommitted.add(path)
        return uncommitted
    except (FileNotFoundError, subprocess.SubprocessError):
        return set()


def _should_skip_test(module_path: str) -> bool:
    """Check if slow test should be skipped because its files are unchanged."""
    deps = _SLOW_TEST_DEPS.get(module_path, [])
    if not deps:
        return False
    uncommitted = _get_uncommitted_files()
    # Run tests if the test file itself is uncommitted
    if module_path in uncommitted:
        return False
    # Run tests if any dependency source file is uncommitted
    for dep in deps:
        if dep in uncommitted:
            return False
        # Check if any file in the dependency directory is uncommitted
        dep_dir = dep.rsplit("/", 1)[0] if "/" in dep else ""
        if dep_dir:
            for u in uncommitted:
                if u.startswith(dep_dir + "/") or u == dep:
                    return False
    return True


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow tests if their source files are unchanged."""
    skip_marker = pytest.mark.skip(
        reason=("Skipping slow test (source files unchanged). Run with --run-slow to force execution."),
    )
    for item in items:
        module_file = getattr(item, "module", None)
        if module_file is None:
            continue
        abs_path = getattr(module_file, "__file__", "") or ""
        try:
            rel_path = os.path.relpath(abs_path, _PROJECT_ROOT)
        except ValueError:
            rel_path = abs_path
        if rel_path in _SLOW_TEST_DEPS:
            if _should_skip_test(rel_path):
                item.add_marker(skip_marker)


@pytest.fixture(autouse=True)
def _track_test_time(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Track and report per-test duration."""
    start = time.monotonic()
    yield
    elapsed = time.monotonic() - start
    test_name = request.node.name
    test_cls = getattr(request.node, "cls", None)
    test_class = test_cls.__name__ if test_cls else ""
    prefix = f"{test_class}::" if test_class else ""
    print(f"\r  [time={elapsed:.3f}s] {prefix}{test_name}", end="", flush=True)


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: Any,
) -> None:
    """Print summary of skipped slow tests."""
    skipped_list = terminalreporter.stats.get("skipped", [])
    skipped_slow: list[Any] = []
    for rep in skipped_list:
        reason = str(getattr(rep, "longrepr", ""))
        if "source files unchanged" in reason:
            skipped_slow.append(rep)
    if skipped_slow:
        terminalreporter.write_sep("=", "SLOW TESTS SKIPPED (source unchanged)")
        for rep in skipped_slow:
            msg = "skipped"
            if hasattr(rep, "longrepr") and rep.longrepr:
                lines = str(rep.longrepr).splitlines()
                if lines:
                    msg = lines[-1]
            terminalreporter.write_line(f"  {rep.nodeid}: {msg}")
