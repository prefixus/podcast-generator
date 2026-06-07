"""Text-to-speech script builder for podcast creation.

Converts extracted PDF sections into a structured, TTS-ready script
suitable for popular providers (ElevenLabs, Amazon Polly, Azure, Google).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from preprocess.pdf_extractor import DocumentStructure, Section


@dataclass
class TTSChunk:
    """A single chunk of text optimized for TTS processing."""

    id: str
    text: str
    section_number: int
    section_title: str
    is_intro: bool = False
    is_outro: bool = False
    is_transition: bool = False


@dataclass
class PodcastScript:
    """Complete podcast script ready for TTS API calls."""

    title: str
    chunks: list[TTSChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    """Clean text for TTS: remove artifacts, normalize spacing, expand abbreviations."""
    # Remove bullet point markers (●) and replace with clean text
    text = text.replace("● ", "– ")

    # Remove section number prefixes from body text (e.g., "1. Zgwałcenie" mid-paragraph)
    # but keep them for section headers
    text = re.sub(r"(?<=\s)\d+\.\s+(?=[A-ZĘÓĄŚŁŻŃĆŻ])", "", text)

    # Remove trailing arrows and references like "→ art. 197 k.k."
    # Keep them but clean up
    text = re.sub(r"\s+→\s+art\.\s+\d+\s+k\.\s*[k\.w\.]?", "", text)

    # Normalize multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Remove trailing/leading whitespace
    text = text.strip()

    # Handle common Polish TTS issues
    # Expand "np." to "na przykład" for TTS clarity
    text = re.sub(r"\bnp\.", "na przykład", text)

    # Expand "czyli" stays as is (fine for TTS)
    # Expand "tzw." to "tak zwany"
    text = re.sub(r"\btzw\.", "tak zwany", text)

    # Handle roman numerals in context (k.k. = kodeks karny, k.w. = kodeks wykroczeń)
    text = re.sub(r"\bk\.k\.", "kodeks karny", text)
    text = re.sub(r"\bk\.w\.", "kodeks wykroczeń", text)

    # Handle year references like "2014 roku" - keep as is (TTS handles numbers)
    # Handle "ICD-11" and "DSM-V" - keep as is (TTS handles acronyms)
    # Handle "GHB", "ketamina" etc. - keep as is

    # Remove page break artifacts and orphaned numbers
    text = re.sub(r"\b\d+\.\s*$", "", text)

    # Remove stray page markers like "Page X" patterns
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)

    return text.strip()


def _build_introduction(doc: DocumentStructure) -> TTSChunk:
    """Build podcast introduction chunk."""
    intro_text = (
        f"Witajcie w odcinku podcastu poświęconemu tematowi: "
        f"{doc.title}. "
        f"To jest cykl {len(doc.sections)} lektów, obejmujący "
        f"całą tematykę prezentowaną w materiale źródłowym. "
        f"Przejdźmy do pierwszego wątku."
    )
    return TTSChunk(
        id="intro",
        text=_normalize_text(intro_text),
        section_number=0,
        section_title="Wstęp",
        is_intro=True,
    )


def _build_outro(doc: DocumentStructure) -> TTSChunk:
    """Build podcast outro chunk."""
    outro_text = (
        f"To był nasz przegląd tematów związanych z "
        f"{doc.title}. "
        f"Dziękujemy za uwagę i do usłyszenia w następnym odcinku."
    )
    return TTSChunk(
        id="outro",
        text=_normalize_text(outro_text),
        section_number=len(doc.sections) + 1,
        section_title="Zakończenie",
        is_outro=True,
    )


def _build_transitions(sections: list[Section]) -> list[TTSChunk]:
    """Build transition text between sections."""
    transitions: list[TTSChunk] = []
    for i, section in enumerate(sections):
        if i == 0:
            continue
        prev = sections[i - 1]
        # Create a natural-sounding transition
        transition_text = (
            f"Przechodzimy do kolejnego tematu. "
            f"W poprzednim odcinku poruszyliśmy sprawę "
            f"{prev.title.lower()}. "
            f"Teraz zajmiemy się tematem: {section.title}."
        )
        transitions.append(
            TTSChunk(
                id=f"transition_{section.number}",
                text=_normalize_text(transition_text),
                section_number=section.number,
                section_title=f"Przejście: {section.title}",
                is_transition=True,
            )
        )
    return transitions


def _chunk_body_for_tts(body: str, max_chars: int = 3000) -> list[str]:
    """Split body text into TTS-friendly chunks respecting sentence boundaries."""
    if not body:
        return []

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", body)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Sentence-specific cleanup
        sentence = re.sub(r"\s{2,}", " ", sentence)

        if current_len + len(sentence) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_len = len(sentence)
        else:
            current_chunk.append(sentence)
            current_len += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def build_podcast_script(doc: DocumentStructure) -> PodcastScript:
    """Build a complete TTS-ready podcast script from parsed document."""
    chunks: list[TTSChunk] = []

    # Add introduction
    chunks.append(_build_introduction(doc))

    # Process each section
    for section in doc.sections:
        # Add transition (except before first section)
        if section.number > 1:
            # Find the right transition
            prev_section = None
            for s in doc.sections:
                if s.number == section.number - 1:
                    prev_section = s
                    break
            if prev_section:
                transition_text = (
                    f"Przechodzimy do kolejnego tematu. "
                    f"W poprzednim odcinku poruszyliśmy sprawę "
                    f"{prev_section.title.lower()}. "
                    f"Teraz zajmiemy się tematem: {section.title}."
                )
                chunks.append(
                    TTSChunk(
                        id=f"transition_{section.number}",
                        text=_normalize_text(transition_text),
                        section_number=section.number,
                        section_title=f"Przejście: {section.title}",
                        is_transition=True,
                    )
                )

        # Add section title (spoken clearly)
        title_chunk = TTSChunk(
            id=f"title_{section.number}",
            text=f"Temat numer {section.number}: {section.title}.",
            section_number=section.number,
            section_title=section.title,
        )
        chunks.append(title_chunk)

        # Split body into TTS-friendly chunks
        body_chunks = _chunk_body_for_tts(section.body)
        for chunk_idx, body_chunk in enumerate(body_chunks):
            cleaned = _normalize_text(body_chunk)
            if cleaned:
                chunks.append(
                    TTSChunk(
                        id=f"chunk_{section.number}_{chunk_idx}",
                        text=cleaned,
                        section_number=section.number,
                        section_title=section.title,
                    )
                )

    # Add outro
    chunks.append(_build_outro(doc))

    return PodcastScript(
        title=doc.title,
        chunks=chunks,
        metadata={
            "total_sections": len(doc.sections),
            "total_chunks": len(chunks),
            "total_characters": sum(len(c.text) for c in chunks),
        },
    )


def save_script_json(script: PodcastScript, output_path: str | Path) -> Path:
    """Save podcast script as JSON for TTS API processing."""
    import json

    data = {
        "title": script.title,
        "metadata": script.metadata,
        "chunks": [
            {
                "id": c.id,
                "text": c.text,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "is_intro": c.is_intro,
                "is_outro": c.is_outro,
                "is_transition": c.is_transition,
            }
            for c in script.chunks
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return output


def save_script_text(script: PodcastScript, output_path: str | Path) -> Path:
    """Save podcast script as plain text for TTS processing."""
    lines: list[str] = []

    for chunk in script.chunks:
        if chunk.is_intro:
            lines.append("[INTRO]")
        elif chunk.is_outro:
            lines.append("[OUTRO]")
        elif chunk.is_transition:
            lines.append("[TRANSITION]")
        elif chunk.section_title and chunk.id.startswith("title_"):
            lines.append(f"[SECTION {chunk.section_number}] {chunk.section_title}")
        else:
            lines.append(chunk.text)

        lines.append("")  # blank line between chunks

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    return output
