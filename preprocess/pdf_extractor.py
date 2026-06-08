"""PDF text extraction with cleanup for TTS-ready podcast preparation.

Handles complex PDF layouts with numbered theses, bullet points,
and cross-page sections. Only sections with substantial body text
are kept (summary pages with short list items are filtered out).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Section:
    """A single numbered thesis/section from the document."""

    number: int
    title: str
    body: str
    page: int = 0


@dataclass
class DocumentStructure:
    """Parsed document with sections and metadata."""

    title: str
    sections: list[Section] = field(default_factory=list)
    raw_full_text: str = ""


def _extract_title(text: str) -> str:
    """Extract document title from the first meaningful line."""
    lines = text.strip().split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 10:
            return stripped
    return "Unknown Document"


def _clean_line(text: str) -> str:
    """Normalize spacing and line breaks within a single line."""
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _is_section_header(line: str) -> bool:
    """Check if a line is a numbered section header.

    Excludes lines that contain multiple numbered items (summary pages)
    like '4. Item 5. Another 6. Yet another'.
    """
    # Count ALL numbered items in this line (not just at start)
    matches = re.findall(r"\b\d+\.\s+", line)
    if len(matches) > 1:
        # Multiple numbers on one line = summary page, not a real section
        return False
    return bool(re.match(r"^\d+\.\s+[A-ZĘÓĄŚŁŻŃĆŻ]", line))


def _is_summary_item(line: str) -> bool:
    """Check if a line is a short summary list item (legal reference).

    Summary items are short, contain "→" arrows, or reference articles.
    """
    # Short lines with legal references
    if "→ art." in line or "→" in line:
        if len(line) < 150:
            return True
    # Lines that are just "Art. XXX" references
    if re.match(r"^Art\.\s+\d+\s+→", line):
        return True
    # Very short numbered items that are clearly list entries
    match = re.match(r"^\d+\.\s+(.+)$", line)
    if match:
        text = match.group(1)
        if len(text) < 80 and ("→" in text or "art." in text.lower()):
            return True
    return False


def _is_summary_page_header(line: str) -> bool:
    """Check if a line is a summary/list page header."""
    patterns = [
        r"Najważniejsze\s+przykłady:",
        r"Zgwałcenia\s+motywowane\s+głównie",
        r"Zgwałcenia\s+motywowane\s+niespecyficznie",
        r"Zgwałcenia\s+psychopatyczne",
        r"Zgwałcenia\s+impulsywne",
        r"Kodeks\s+wykroczeń",
        r"Rozdział\s+XVI",
    ]
    for pattern in patterns:
        if re.search(pattern, line):
            return True
    return False


def _has_substantial_body(body: str, min_words: int = 100) -> bool:
    """Check if body text is substantial enough to be a real section."""
    words = body.split()
    return len(words) >= min_words


def _is_real_section_title(title: str) -> bool:
    """Check if a title looks like a real thesis title, not a summary list item.

    Summary items tend to be short, contain legal references (→ art.),
    or are truncated (ending with semicolons, dashes, or incomplete thoughts).
    """
    # Reject titles that are clearly legal reference summaries
    if "→ art." in title or "→" in title:
        return False
    # Reject titles that are clearly truncated (end with semicolon, dash, or comma)
    if title.rstrip().endswith((";", "-", ",", "–", "—")):
        return False
    # Real thesis titles are typically longer
    if len(title) < 25:
        return False
    return True


def _extract_sections_from_pages(reader: PdfReader) -> list[Section]:
    """Extract structured sections (numbered theses) from all pages.

    Handles cross-page sections by maintaining state across page boundaries.
    Only sections with substantial body text are kept (summary page items
    with short legal references are filtered out).
    """
    all_sections: list[Section] = []
    seen_titles: set[str] = set()

    current_section: Section | None = None

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        lines = text.split("\n")

        # Clean all lines and filter empty
        cleaned_lines: list[str] = []
        for line in lines:
            stripped = _clean_line(line)
            if stripped:
                cleaned_lines.append(stripped)

        if not cleaned_lines:
            continue

        for line in cleaned_lines:
            # Check for section header (numbered line starting with digit + period + capital)
            if _is_section_header(line):
                # Save previous section (if it had substantial body AND valid title)
                if current_section is not None:
                    if _has_substantial_body(current_section.body) and _is_real_section_title(current_section.title):
                        if current_section.title not in seen_titles:
                            seen_titles.add(current_section.title)
                            all_sections.append(current_section)

                match = re.match(r"^(\d+)\.\s+(.+)$", line)
                if match:
                    sec_num = int(match.group(1))
                    sec_title = match.group(2).strip()
                    current_section = Section(
                        number=sec_num,
                        title=sec_title,
                        body="",
                        page=page_idx + 1,
                    )
                continue

            # Skip summary page headers
            if _is_summary_page_header(line):
                continue

            # Check for bullet-point style list items ("– something")
            if line.startswith("– ") or line.startswith("• "):
                if current_section is not None:
                    if current_section.body:
                        current_section.body += " " + line
                    else:
                        current_section.body = line
                continue

            # Check for "Art." references (summary list items)
            if re.match(r"^Art\.\s+\d+", line):
                if current_section is not None:
                    if current_section.body:
                        current_section.body += " " + line
                    else:
                        current_section.body = line
                continue

            # Check for short summary items (legal references with arrows)
            if _is_summary_item(line):
                continue

            # If we haven't started the body yet, check if this line continues the title
            if current_section is not None and not current_section.body:
                title_words = current_section.title.split()
                last_word = title_words[-1].lower() if title_words else ""
                ends_with_dash = (
                    current_section.title.rstrip().endswith(("-", "–", "—", ","))
                    or last_word in ("i", "lub", "a", "oraz")
                )
                starts_with_lower = bool(re.match(r"^[a-zęóąśłżńćż]", line))
                is_article = bool(re.match(r"^(art\b|ust\b|par\b|\d+)", line, re.IGNORECASE))
                if (starts_with_lower or ends_with_dash) and not is_article:
                    current_section.title += " " + line
                    continue

            # Regular body text
            if current_section is not None:
                if current_section.body:
                    current_section.body += " " + line
                else:
                    current_section.body = line

    # Save last section (only if substantial AND valid title)
    if current_section is not None:
        if _has_substantial_body(current_section.body) and _is_real_section_title(current_section.title):
            if current_section.title not in seen_titles:
                seen_titles.add(current_section.title)
                all_sections.append(current_section)

    return all_sections


def extract_and_parse(pdf_path: str | Path) -> DocumentStructure:
    """Extract text from PDF, clean it, and structure it into sections."""
    reader = PdfReader(str(pdf_path))

    # Build full raw text for reference
    full_parts: list[str] = []
    for page in reader.pages:
        full_parts.append(page.extract_text())
    raw_full_text = "\n".join(full_parts)

    # Extract structured sections (cross-page aware, filters summary items)
    sections = _extract_sections_from_pages(reader)

    # Extract title from first page
    first_text = reader.pages[0].extract_text()
    title = _extract_title(first_text)

    return DocumentStructure(title=title, sections=sections, raw_full_text=raw_full_text)
