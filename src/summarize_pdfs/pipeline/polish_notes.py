from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

from rich.console import Console

from summarize_pdfs.config import AppConfig, Settings
from summarize_pdfs.export.dedupe_notes import deduplicate_study_notes
from summarize_pdfs.export.plaintext import sanitize_plaintext
from summarize_pdfs.export.summary import TOPIC_ORDER
from summarize_pdfs.pipeline.llm import LLMClient, chat_json, coerce_json_dict, make_llm_client, map_concurrent
from summarize_pdfs.pipeline.prompts import POLISH_NOTES_SYSTEM, polish_notes_prompt

console = Console()

_TOPIC_HEADER_RE = re.compile(r"^(.+?) — Quick Notes\s*$")
_BULLET_START_RE = re.compile(r"^(?:•|\s+where\b)", re.I)
_WHERE_LINE_RE = re.compile(r"^\s*where\b", re.I)
_SECTION_HEADER_RE = re.compile(r"^(?:Key Facts|Formulas):\s*$", re.I)


def _extract_polished_text(data: object) -> str:
    data = coerce_json_dict(data)
    if not data:
        return ""
    for key in ("polished_text", "text", "content"):
        value = data.get(key)
        if value:
            return str(value)
    for key, value in data.items():
        normalized = key.replace("\\", "").lower()
        if "polished" in normalized and value:
            return str(value)
    return ""


def _normalize_section_bullets(text: str) -> str:
    """Ensure consistent • bullets and indented where-clauses."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _TOPIC_HEADER_RE.match(stripped):
            lines.append(stripped)
            continue
        if _SECTION_HEADER_RE.match(stripped):
            lines.append(stripped)
            continue
        if _WHERE_LINE_RE.match(stripped):
            lines.append(f"  {stripped}")
            continue
        if stripped.startswith("•"):
            lines.append(stripped if stripped.startswith("• ") else "• " + stripped[1:].lstrip())
            continue
        if stripped.startswith(("Def:", "Trick:", "Fix:")):
            lines.append(f"• {stripped}")
            continue
        if lines and _WHERE_LINE_RE.match(lines[-1].strip()):
            lines[-1] = f"{lines[-1]} {stripped}"
        else:
            lines.append(f"• {stripped}")
    return "\n".join(lines)


def _short_topic_label(topic: str) -> str:
    return topic.split(" (")[0].upper()


def _match_topic(header_label: str) -> str:
    label = header_label.strip().upper()
    for topic in TOPIC_ORDER:
        short = _short_topic_label(topic)
        if label == short or label.startswith(short) or short.startswith(label):
            return topic
    return header_label.strip()


def split_study_notes_by_topic(notes_text: str) -> tuple[str, dict[str, str]]:
    """Split study notes into document header and topic sections."""
    lines = notes_text.splitlines()
    doc_header_lines: list[str] = []
    sections: dict[str, str] = {}
    current_topic: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = _TOPIC_HEADER_RE.match(line)
        if match:
            if current_topic is not None:
                sections[current_topic] = "\n".join(current_lines).strip()
            current_topic = _match_topic(match.group(1))
            current_lines = [line]
            continue
        if current_topic is None:
            doc_header_lines.append(line)
        else:
            current_lines.append(line)

    if current_topic is not None:
        sections[current_topic] = "\n".join(current_lines).strip()

    preamble: list[str] = []
    orphan_content: list[str] = []
    for line in doc_header_lines:
        if _BULLET_START_RE.match(line):
            orphan_content.append(line)
        elif orphan_content:
            orphan_content.append(line)
        else:
            preamble.append(line)

    if orphan_content and "Descriptive Statistics" not in sections:
        label = _short_topic_label("Descriptive Statistics")
        sections["Descriptive Statistics"] = (
            f"{label} — Quick Notes\n\n" + "\n".join(orphan_content).strip()
        )

    doc_header = "\n".join(preamble).strip()
    return doc_header, sections


def render_polished_notes(doc_header: str, polished: dict[str, str]) -> str:
    lines: list[str] = []
    if doc_header:
        lines.append(doc_header)
        lines.append("")

    seen: set[str] = set()
    for topic in TOPIC_ORDER:
        if topic in polished and topic not in seen:
            lines.append(polished[topic].rstrip())
            lines.append("")
            seen.add(topic)

    for topic, text in polished.items():
        if topic not in seen:
            lines.append(text.rstrip())
            lines.append("")
            seen.add(topic)

    return "\n".join(lines).rstrip() + "\n"


async def _polish_topic(
    topic: str,
    section_text: str,
    *,
    client: LLMClient,
    config: AppConfig,
) -> tuple[str, str]:
    prompt = polish_notes_prompt(section_text, topic)
    data = await chat_json(
        client,
        model=config.llm.model,
        prompt=prompt,
        temperature=config.llm.temperature,
        cache_dir=config.cache_dir / "polish_notes",
        system_prompt=POLISH_NOTES_SYSTEM,
    )
    polished = sanitize_plaintext(_extract_polished_text(data)).strip()
    if not polished:
        console.print(f"[yellow]Polish returned empty for {topic} — keeping original[/yellow]")
        return topic, section_text
    polished = _normalize_section_bullets(polished)
    return topic, polished


async def polish_study_notes(
    config: AppConfig,
    *,
    notes_path: Path | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Polish study notes by topic with concurrent LLM enhancement."""
    settings = Settings()
    client = make_llm_client(config, settings)

    notes_path = notes_path or config.output_dir / "study_notes.txt"
    if not notes_path.exists():
        raise FileNotFoundError(f"Study notes not found: {notes_path}")

    notes_text = notes_path.read_text()
    doc_header, sections = split_study_notes_by_topic(notes_text)
    if not sections:
        raise RuntimeError(f"No topic sections found in {notes_path}")

    topics = [t for t in TOPIC_ORDER if t in sections]
    topics.extend(t for t in sections if t not in topics)

    console.print(f"Polishing {len(topics)} topic sections (max {config.llm.max_concurrent} concurrent)...")
    coros = [
        _polish_topic(topic, sections[topic], client=client, config=config)
        for topic in topics
    ]
    results = await map_concurrent(coros, limit=config.llm.max_concurrent)
    polished = {topic: text for topic, text in results}

    if output_path is None:
        output_path = config.output_dir / ("study_notes.txt" if overwrite else "study_notes_polished.txt")

    output_text = deduplicate_study_notes(render_polished_notes(doc_header, polished))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and notes_path.exists():
        backup = notes_path.with_suffix(notes_path.suffix + ".bak")
        shutil.copy2(notes_path, backup)
        console.print(f"[dim]Backup → {backup}[/dim]")

    output_path.write_text(output_text)
    console.print(f"[green]Polished notes[/green] → {output_path}")
    return output_path


def polish_study_notes_sync(
    config: AppConfig,
    *,
    notes_path: Path | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    return asyncio.run(
        polish_study_notes(
            config,
            notes_path=notes_path,
            output_path=output_path,
            overwrite=overwrite,
        )
    )
