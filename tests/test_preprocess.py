"""Tests for the PDF-to-podcast preprocessor pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from preprocess.pdf_extractor import DocumentStructure, Section, extract_and_parse
from preprocess.tts_script_builder import (
    PodcastScript,
    build_podcast_script,
    save_script_json,
    save_script_text,
)

EXAMPLE_PDF = "example-data/Seksuologia_opracowane_tezy.pdf"
OUTPUT_DIR = "tests/output"


class TestExtractAndParse:
    """Tests for PDF extraction and parsing."""

    def test_extract_returns_document_structure(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        assert isinstance(doc, DocumentStructure)
        assert doc.title

    def test_sections_have_content(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        assert len(doc.sections) > 50  # Should have many sections
        for section in doc.sections:
            assert section.number > 0
            assert section.title
            assert len(section.body.split()) >= 100  # substantial body

    def test_sections_have_valid_titles(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        for section in doc.sections:
            assert "→ art." not in section.title  # no legal refs in titles
            assert not section.title.rstrip().endswith((";", "-", "–"))  # not truncated

    def test_raw_full_text_exists(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        assert len(doc.raw_full_text) > 1000

    def test_summary_items_filtered(self) -> None:
        """Summary page items (short, legal references) should be filtered."""
        doc = extract_and_parse(EXAMPLE_PDF)
        for section in doc.sections:
            assert "→ art." not in section.title


class TestBuildPodcastScript:
    """Tests for TTS script building."""

    def test_script_has_intro(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        intro = next(c for c in script.chunks if c.id == "intro")
        assert intro.is_intro is True
        assert "Witajcie" in intro.text

    def test_script_has_outro(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        outro = next(c for c in script.chunks if c.id == "outro")
        assert outro.is_outro is True

    def test_script_has_transitions(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        transitions = [c for c in script.chunks if c.is_transition]
        assert len(transitions) > 0

    def test_script_has_section_titles(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        titles = [c for c in script.chunks if c.id.startswith("title_")]
        assert len(titles) == len(doc.sections)

    def test_script_metadata(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        assert script.metadata["total_sections"] == len(doc.sections)
        assert script.metadata["total_chunks"] == len(script.chunks)
        assert script.metadata["total_characters"] > 0

    def test_body_chunks_exist(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        body_chunks = [c for c in script.chunks if not c.is_intro and not c.is_outro and not c.is_transition and not c.id.startswith("title_")]
        assert len(body_chunks) > 0


class TestNormalizeText:
    """Tests for text normalization."""

    def test_expand_np(self) -> None:
        from preprocess.tts_script_builder import _normalize_text
        result = _normalize_text("np.")
        assert result == "na przykład"

    def test_expand_kk(self) -> None:
        from preprocess.tts_script_builder import _normalize_text
        result = _normalize_text("art. 197 kodeks karny")
        assert "kodeks karny" in result

    def test_expand_tzw(self) -> None:
        from preprocess.tts_script_builder import _normalize_text
        result = _normalize_text("tzw.")
        assert result == "tak zwany"

    def test_normalize_spacing(self) -> None:
        from preprocess.tts_script_builder import _normalize_text
        result = _normalize_text("test    multiple    spaces")
        assert "test multiple spaces" == result


class TestSaveScript:
    """Tests for script output saving."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self) -> None:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def test_save_json(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        output_path = save_script_json(script, Path(OUTPUT_DIR) / "test_script.json")
        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "title" in data
        assert "chunks" in data
        assert "metadata" in data
        assert len(data["chunks"]) > 0

    def test_save_text(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        output_path = save_script_text(script, Path(OUTPUT_DIR) / "test_script.txt")
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "[INTRO]" in content
        assert "[OUTRO]" in content
        assert "[SECTION" in content

    def test_json_chunk_structure(self) -> None:
        doc = extract_and_parse(EXAMPLE_PDF)
        script = build_podcast_script(doc)
        output_path = save_script_json(script, Path(OUTPUT_DIR) / "test_chunk.json")
        data = json.loads(output_path.read_text(encoding="utf-8"))
        chunk = data["chunks"][0]
        assert "id" in chunk
        assert "text" in chunk
        assert "section_number" in chunk
        assert "section_title" in chunk
        assert "is_intro" in chunk
        assert "is_outro" in chunk
        assert "is_transition" in chunk
