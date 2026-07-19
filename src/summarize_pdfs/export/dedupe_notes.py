from __future__ import annotations

import re

from summarize_pdfs.export.summary import TOPIC_ORDER

_TOPIC_HEADER_RE = re.compile(r"^(.+?) — Quick Notes\s*$")
_SECTION_HEADER_RE = re.compile(r"^(?:Key Facts|Formulas):\s*$", re.I)
_WHERE_LINE_RE = re.compile(r"^\s*where\b", re.I)

_JUNK_LINE_RE = re.compile(
    r"(?:"
    r"label:\s*correct formula|exam-solving tip without worked numbers|"
    r"Explain errors in existing notes|where var = meaning|"
    r"Histograms, Frequency Polygons, and Time Series Graphs: Chapter|"
    r"if you have a sample of \d+ numbers:"
    r")",
    re.IGNORECASE,
)

_INCOMPLETE_WHERE_RE = re.compile(
    r"^\s*where\b.{0,40}$|;\s*P\s*$|;\s*P\([A-Z]\|B\)\s*$",
    re.IGNORECASE,
)


def _normalize_key(text: str) -> str:
    text = re.sub(r"^[•\s]+", "", text.strip())
    text = re.sub(r"^(Def|Trick|Fix):\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).lower()
    # For formulas, key on lhs or full expression before long prose
    if "=" in text and not text.lower().startswith("def:"):
        lhs = text.split("=", 1)[0].strip()
        if len(lhs) < 40:
            return lhs
    return text[:120]


def _is_junk_block(bullet: str, where: str | None) -> bool:
    combined = f"{bullet} {where or ''}"
    if _JUNK_LINE_RE.search(combined):
        return True
    if where and _INCOMPLETE_WHERE_RE.match(where):
        return True
    if bullet.strip().endswith(":") and len(bullet) < 20:
        return True
    return False


def _parse_blocks(lines: list[str]) -> list[tuple[str, str | None]]:
    blocks: list[tuple[str, str | None]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or _TOPIC_HEADER_RE.match(line):
            i += 1
            continue
        if line.startswith("•"):
            bullet = line
            where: str | None = None
            if i + 1 < len(lines) and _WHERE_LINE_RE.match(lines[i + 1]):
                where = lines[i + 1]
                i += 2
            else:
                i += 1
            if not _is_junk_block(bullet, where):
                blocks.append((bullet, where))
            continue
        i += 1
    return blocks


def _render_blocks(blocks: list[tuple[str, str | None]]) -> list[str]:
    lines: list[str] = []
    for bullet, where in blocks:
        lines.append(bullet)
        if where:
            lines.append(where)
    return lines


def _dedupe_block_list(
    blocks: list[tuple[str, str | None]],
    *,
    seen: set[str] | None = None,
    global_formula_keys: set[str] | None = None,
) -> list[tuple[str, str | None]]:
    local_seen = seen if seen is not None else set()
    kept: list[tuple[str, str | None]] = []
    for bullet, where in blocks:
        key = _normalize_key(bullet)
        if where:
            key = f"{key}|{_normalize_key(where)[:80]}"
        if key in local_seen:
            continue
        is_formula = "=" in bullet and not re.search(r"^•\s*Def:", bullet, re.I)
        if global_formula_keys is not None and is_formula and key in global_formula_keys:
            continue
        local_seen.add(key)
        if global_formula_keys is not None and is_formula:
            global_formula_keys.add(key)
        kept.append((bullet, where))
    return kept


def deduplicate_within_section(
    lines: list[str],
    *,
    global_formula_keys: set[str] | None = None,
) -> list[str]:
    """Dedupe bullets within a topic section, preserving Key Facts:/Formulas: subheaders."""
    result: list[str] = []
    trailing: list[str] = []
    i = 0

    def append_deduped(chunk: list[str]) -> None:
        if not chunk:
            return
        deduped = _render_blocks(
            _dedupe_block_list(_parse_blocks(chunk), global_formula_keys=global_formula_keys)
        )
        if deduped:
            result.extend(deduped)

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if _SECTION_HEADER_RE.match(stripped):
            append_deduped(trailing)
            trailing = []
            result.append(stripped)
            result.append("")
            i += 1
            chunk: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if _SECTION_HEADER_RE.match(nxt.strip()) or _TOPIC_HEADER_RE.match(nxt):
                    break
                chunk.append(nxt)
                i += 1
            append_deduped(chunk)
            continue
        if _TOPIC_HEADER_RE.match(line):
            break
        if stripped:
            trailing.append(line)
        i += 1

    append_deduped(trailing)
    return result


def deduplicate_study_notes(text: str, *, global_formulas: bool = True) -> str:
    """Remove duplicate bullets and junk; optionally dedupe formulas across topics."""
    lines = text.splitlines()
    doc_header: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in lines:
        match = _TOPIC_HEADER_RE.match(line)
        if match:
            current = line.replace(" — Quick Notes", "").strip()
            for topic in TOPIC_ORDER:
                if topic.upper().startswith(current.upper()[:20]) or current.upper().startswith(
                    topic.split(" (")[0].upper()[:20]
                ):
                    current = topic
                    break
            sections.setdefault(current, [line])
            continue
        if current is None:
            doc_header.append(line)
        else:
            sections[current].append(line)

    global_formula_keys: set[str] = set()
    polished_sections: dict[str, list[str]] = {}

    for topic in TOPIC_ORDER:
        if topic not in sections:
            continue
        section_lines = sections[topic]
        header = section_lines[0] if section_lines else f"{topic.upper()} — Quick Notes"
        formula_keys = global_formula_keys if global_formulas else None
        body = deduplicate_within_section(section_lines[1:], global_formula_keys=formula_keys)
        polished_sections[topic] = [header, ""] + body

    out: list[str] = []
    if doc_header:
        out.extend(doc_header)
        out.append("")
    for topic in TOPIC_ORDER:
        if topic in polished_sections:
            out.extend(polished_sections[topic])
            out.append("")
    for topic, content in polished_sections.items():
        if topic not in TOPIC_ORDER:
            out.extend(content)
            out.append("")

    return "\n".join(out).strip() + "\n"
