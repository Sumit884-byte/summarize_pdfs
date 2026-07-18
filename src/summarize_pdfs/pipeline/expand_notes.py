from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console

from summarize_pdfs.config import AppConfig, Settings
from summarize_pdfs.export.formula_glossary import render_formula_lines
from summarize_pdfs.export.topic_facts import canonical_facts_for_topic
from summarize_pdfs.export.plaintext import sanitize_plaintext
from summarize_pdfs.export.notes import export_notes_file
from summarize_pdfs.export.summary import TOPIC_ORDER, export_summary_file
from summarize_pdfs.index.vector_store import VectorStore
from summarize_pdfs.pipeline.llm import LLMClient, chat_json, make_llm_client
from summarize_pdfs.pipeline.prompts import SYSTEM_EXPAND, expand_topic_user_prompt

console = Console()

_TOPIC_QUERIES: dict[str, list[str]] = {
    "Descriptive Statistics": [
        "mean median mode variance standard deviation",
        "measures of center and spread sample statistics",
        "quartiles percentiles box plot outliers",
        "key facts properties mean sensitive outliers median robust",
        "sample vs population descriptive statistics important notes",
    ],
    "Probability & Conditional Probability": [
        "probability rules conditional probability independence",
        "Bayes theorem prior posterior probability",
        "complement rule at least one event",
        "key facts probability range 0 to 1 mutually exclusive independent",
        "properties conditional probability sample space",
    ],
    "Combinatorics & Counting": [
        "permutations combinations factorial counting",
        "multiplication rule arrangements with restrictions",
        "key facts order matters permutation combination replacement",
        "properties counting with without replacement",
    ],
    "Correlation & Association": [
        "correlation coefficient scatter plot association",
        "linear correlation Pearson r",
        "key facts correlation causation association strength direction",
        "properties contingency table categorical variables",
    ],
    "Data Types & Study Design": [
        "nominal ordinal interval ratio data types",
        "sampling methods random sample population",
        "cross sectional time series study design",
        "key facts measurement scales nominal ordinal interval ratio",
        "properties study design sampling bias",
    ],
    "Frequency & Distribution": [
        "frequency distribution histogram cumulative frequency",
        "IQR interquartile range quartiles",
        "key facts relative frequency cumulative distribution outliers",
        "properties histogram distribution shape",
    ],
    "Transformations of Data": [
        "linear transformation mean variance standard deviation",
        "effect of adding constant multiplying data",
        "key facts linear transform mean standard deviation shift scale",
        "properties adding constant multiplying data spread",
    ],
    "Exam Skills (MCQ / MSQ / SA)": [
        "interpreting multiple choice statistics questions",
        "choosing correct statistical method",
        "key facts exam strategies complement rule at least one",
    ],
}


def _retrieve_topic_chunks(
    store: VectorStore,
    config: AppConfig,
    topic: str,
) -> list[dict]:
    queries = _TOPIC_QUERIES.get(topic, [topic])
    seen: set[str] = set()
    hits: list[dict] = []
    for query in queries:
        for hit in store.query(query, top_k=config.top_k, doc_type="textbook"):
            chunk_id = hit["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            hits.append(hit)
            if len(hits) >= config.max_chunks_per_topic:
                break
        if len(hits) >= config.max_chunks_per_topic:
            break
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def _format_hits(hits: list[dict]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata") or {}
        page = meta.get("page", "?")
        source = Path(meta.get("source_path", "textbook")).name
        blocks.append(f"[{i}] p.{page} ({source})\n{hit['text']}")
    return "\n\n".join(blocks)


def _existing_topic_notes(notes_text: str, topic: str) -> str:
    label = topic.upper()
    if " (" in topic:
        label = topic.split(" (")[0].upper()
    lines: list[str] = []
    capture = False
    for line in notes_text.splitlines():
        if line.endswith("— Quick Notes") and label in line.upper():
            capture = True
            lines.append(line)
            continue
        if capture and line.endswith("— Quick Notes") and label not in line.upper():
            break
        if capture:
            lines.append(line)
    return "\n".join(lines).strip()


async def _expand_topic(
    topic: str,
    existing_notes: str,
    hits: list[dict],
    *,
    client: LLMClient,
    config: AppConfig,
    textbook_name: str,
) -> dict:
    prompt = expand_topic_user_prompt(
        topic,
        existing_notes,
        hits,
        textbook_name=textbook_name,
    )
    return await chat_json(
        client,
        model=config.llm.model,
        prompt=prompt,
        temperature=config.llm.temperature,
        cache_dir=None,
        system_prompt=SYSTEM_EXPAND,
    )


def _render_expanded_notes_header(textbook_name: str, chunk_count: int) -> list[str]:
    return [
        "STATISTICS FOR DATA SCIENCE — QUICK STUDY NOTES",
        "=" * 48,
        "",
        f"Expanded with textbook: {textbook_name} ({chunk_count} chunks indexed).",
        "Formulas and definitions cite textbook page numbers. Exam tricks preserved.",
        "",
    ]


def _as_dict_items(value) -> list[dict]:
    if not value:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        items: list[dict] = []
        for item in value:
            if isinstance(item, dict):
                items.append(item)
            elif isinstance(item, str) and item.strip():
                items.append({"text": item})
        return items
    if isinstance(value, str) and value.strip():
        return [{"text": value}]
    return []


def _as_str_items(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("expression") or item.get("name") or ""
                if text:
                    out.append(str(text))
        return out
    return [str(value)]


def _formula_text(item: dict) -> str:
    expr = sanitize_plaintext(item.get("expression", "") or item.get("text", ""))
    name = sanitize_plaintext(item.get("name", ""))
    where = sanitize_plaintext(str(item.get("where_clause") or item.get("where") or ""))
    if name and expr and "=" not in expr:
        line = f"{name}: {expr}"
    else:
        line = expr
    if where:
        line = f"{line}, where {where}"
    return line


def _render_expanded_topic(topic: str, data: dict) -> list[str]:
    data = _normalize_expansion(data)
    label = topic.split(" (")[0].upper() if " (" in topic else topic.upper()
    lines = [f"{label} — Quick Notes", ""]

    seen_fact_keys: set[str] = set()
    topic_facts: list[str] = []
    for fact in canonical_facts_for_topic(topic):
        key = fact.strip().lower()[:80]
        if key not in seen_fact_keys:
            seen_fact_keys.add(key)
            topic_facts.append(fact)
    for item in _as_dict_items(data.get("facts")):
        text = sanitize_plaintext(item.get("text", "") or item.get("name", ""))
        page = item.get("page", "")
        suffix = f" [p.{page}]" if page else ""
        if text:
            key = text.strip().lower()[:80]
            if key not in seen_fact_keys:
                seen_fact_keys.add(key)
                topic_facts.append(f"{text.rstrip('.')}{suffix}")
    for fact in _as_str_items(data.get("facts")):
        text = sanitize_plaintext(fact.strip())
        if text:
            key = text.strip().lower()[:80]
            if key not in seen_fact_keys:
                seen_fact_keys.add(key)
                topic_facts.append(text)

    if topic_facts:
        lines.append("Key Facts:")
        for fact in topic_facts:
            lines.append(f"• {fact.rstrip('.')}")
        lines.append("")

    formula_block: list[str] = []
    for item in _as_dict_items(data.get("formulas")):
        expr = _formula_text(item)
        page = item.get("page", "")
        suffix = f" [p.{page}]" if page else ""
        if expr:
            rendered = render_formula_lines(expr)
            if suffix and rendered:
                rendered[0] = f"{rendered[0]}{suffix}"
            formula_block.extend(rendered)

    if formula_block:
        lines.append("Formulas:")
        lines.extend(formula_block)
        lines.append("")

    for item in _as_dict_items(data.get("definitions")):
        text = sanitize_plaintext(item.get("text", "") or item.get("description", ""))
        name = sanitize_plaintext(item.get("name", ""))
        page = item.get("page", "")
        suffix = f" [p.{page}]" if page else ""
        if text:
            line = text if text.lower().startswith("def:") else f"Def: {name}: {text}" if name else f"Def: {text}"
            lines.append(f"• {line}{suffix}")

    for item in _as_dict_items(data.get("tricks")):
        text = sanitize_plaintext(item.get("text", "") or item.get("name", ""))
        page = item.get("page", "")
        suffix = f" [p.{page}]" if page else ""
        if text:
            tip = text if text.lower().startswith("trick:") else f"Trick: {text}"
            lines.append(f"• {tip}{suffix}")

    for correction in _as_str_items(data.get("corrections")):
        lines.append(f"• Fix: {sanitize_plaintext(correction.strip())}")

    lines.append("")
    return lines


def _normalize_expansion(data) -> dict:
    if isinstance(data, dict):
        normalized = dict(data)
        if "summary_paragraph" not in normalized:
            for key, value in data.items():
                if "summary" in key.lower() and isinstance(value, str):
                    normalized["summary_paragraph"] = value
                    break
        return normalized
    if isinstance(data, list) and data and isinstance(data[0], dict):
        merged: dict = {}
        for item in data:
            for key, value in item.items():
                clean_key = key.replace("\\_", "_")
                if clean_key not in merged:
                    merged[clean_key] = value
                elif isinstance(merged[clean_key], list) and isinstance(value, list):
                    merged[clean_key].extend(value)
        return _normalize_expansion(merged)
    return {}


def _render_expanded_summary_header(textbook_name: str) -> list[str]:
    return [
        "STATISTICS FOR DATA SCIENCE — COMPREHENSIVE STUDY SUMMARY",
        "=" * 58,
        "",
        f"This guide synthesizes five Semester I Statistics exam papers, expanded with",
        f"OpenStax textbook content ({textbook_name}).",
        "Content is grouped by statistical theme with textbook page citations.",
        "",
    ]


def _render_expanded_summary_topic(topic: str, data: dict, existing_summary: str) -> list[str]:
    data = _normalize_expansion(data)
    lines = [topic, "-" * len(topic), ""]
    summary = sanitize_plaintext((data.get("summary_paragraph") or "").strip())
    if summary:
        lines.append(summary)
        lines.append("")

    seen_fact_keys: set[str] = set()
    summary_facts: list[str] = []
    for fact in canonical_facts_for_topic(topic):
        key = fact.strip().lower()[:80]
        if key not in seen_fact_keys:
            seen_fact_keys.add(key)
            summary_facts.append(fact)
    for item in _as_dict_items(data.get("facts")):
        text = sanitize_plaintext(item.get("text", "") or item.get("name", ""))
        page = item.get("page", "")
        suffix = f" (p. {page})" if page else ""
        if text:
            key = text.strip().lower()[:80]
            if key not in seen_fact_keys:
                seen_fact_keys.add(key)
                summary_facts.append(f"{text.rstrip('.')}{suffix}.")

    if summary_facts:
        lines.append("Key facts for this topic:")
        lines.append("")
        for fact in summary_facts:
            lines.append(f"  • {fact.rstrip('.')}.")
        lines.append("")

    if _as_dict_items(data.get("definitions")):
        lines.append("Textbook definitions:")
        lines.append("")
        for item in _as_dict_items(data.get("definitions")):
            name = sanitize_plaintext(item.get("name", ""))
            text = sanitize_plaintext(item.get("text", "") or item.get("description", ""))
            page = item.get("page", "")
            suffix = f" (p. {page})" if page else ""
            lines.append(f"  • {name}: {text.rstrip('.')}{suffix}.")
        lines.append("")

    if _as_dict_items(data.get("formulas")):
        lines.append("Key formulas (textbook-backed):")
        lines.append("")
        for item in _as_dict_items(data.get("formulas")):
            expr = _formula_text(item)
            page = item.get("page", "")
            suffix = f" [p. {page}]" if page else ""
            if expr:
                rendered = render_formula_lines(expr, bullet="  • ", indent="    ")
                if suffix and rendered:
                    rendered[0] = f"{rendered[0]}{suffix}"
                lines.extend(rendered)
        lines.append("")

    if _as_dict_items(data.get("tricks")):
        lines.append("Exam tips:")
        lines.append("")
        for item in _as_dict_items(data.get("tricks")):
            text = sanitize_plaintext(item.get("text", "") or item.get("name", ""))
            if text:
                lines.append(f"  • {text.rstrip('.')}.")
        lines.append("")

    if _as_str_items(data.get("corrections")):
        lines.append("Corrections from textbook:")
        lines.append("")
        for correction in _as_str_items(data.get("corrections")):
            lines.append(f"  • {sanitize_plaintext(correction.rstrip('.'))}.")
        lines.append("")

    # Preserve a snippet of exam-derived content if present
    if existing_summary and "Key formulas (verbatim from source material)" in existing_summary:
        start = existing_summary.find("Representative exam scenarios:")
        if start >= 0:
            snippet = existing_summary[start : start + 800].strip()
            lines.append(snippet)
            lines.append("")

    return lines


async def expand_study_notes(
    config: AppConfig,
    *,
    json_path: Path | None = None,
    notes_path: Path | None = None,
    summary_path: Path | None = None,
) -> tuple[Path, Path, int]:
    """Expand study notes and summary using indexed textbook RAG."""
    settings = Settings()
    client = make_llm_client(config, settings)
    store = VectorStore(config)
    chunk_count = store.count()
    if chunk_count == 0:
        raise RuntimeError("No indexed chunks. Run `summarize-pdfs index` first.")

    out_dir = config.output_dir
    json_path = json_path or sorted(out_dir.glob("study_guide_*.json"), reverse=True)[0]
    notes_path = notes_path or out_dir / "study_notes.txt"
    summary_path = summary_path or out_dir / "study_guide_complete.txt"

    questions_path = config.processed_dir / "questions.jsonl"
    baseline_notes_path = out_dir / "study_notes_baseline.txt"
    baseline_summary_path = out_dir / "study_guide_baseline.txt"
    export_notes_file(json_path, baseline_notes_path, questions_path if questions_path.exists() else None)
    export_summary_file(json_path, baseline_summary_path, questions_path if questions_path.exists() else None)

    existing_notes = baseline_notes_path.read_text()
    existing_summary = baseline_summary_path.read_text()

    books = list((config.source_dir / "textbooks").glob("**/*.pdf"))
    textbook_name = books[0].stem if books else "textbook"

    coros = []
    topics: list[str] = []
    for topic in TOPIC_ORDER:
        hits = _retrieve_topic_chunks(store, config, topic)
        if not hits:
            continue
        topics.append(topic)
        coros.append(
            _expand_topic(
                topic,
                _existing_topic_notes(existing_notes, topic),
                hits,
                client=client,
                config=config,
                textbook_name=textbook_name,
            )
        )

    console.print(f"Expanding {len(topics)} topics using {chunk_count} indexed chunks...")
    expanded = await asyncio.gather(*coros)
    by_topic = {topic: _normalize_expansion(data) for topic, data in zip(topics, expanded)}

    notes_lines = _render_expanded_notes_header(textbook_name, chunk_count)
    summary_lines = _render_expanded_summary_header(textbook_name)

    for topic in TOPIC_ORDER:
        topic_notes = _existing_topic_notes(existing_notes, topic)
        topic_summary = _extract_topic_section(existing_summary, topic)
        has_expansion = topic in by_topic and any(
            by_topic[topic].get(k)
            for k in ("facts", "definitions", "formulas", "tricks", "corrections", "summary_paragraph")
        )

        if topic_notes:
            notes_lines.extend(topic_notes.splitlines())
        if has_expansion:
            notes_lines.extend(_render_expanded_topic(topic, by_topic[topic])[2:])

        if not topic_notes and not has_expansion:
            continue
        if not notes_lines[-1] == "":
            notes_lines.append("")

        if has_expansion:
            summary_lines.extend(
                _render_expanded_summary_topic(topic, by_topic[topic], topic_summary)
            )
        elif topic_summary:
            summary_lines.extend(topic_summary.splitlines())
            summary_lines.append("")

    notes_path.write_text("\n".join(notes_lines).rstrip() + "\n")
    summary_path.write_text("\n".join(summary_lines).rstrip() + "\n")

    manifest = out_dir / "expand_notes_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "textbook": textbook_name,
                "chunk_count": chunk_count,
                "topics_expanded": topics,
                "expansions": by_topic,
            },
            indent=2,
            default=str,
        )
    )
    console.print(f"[green]Expanded notes[/green] → {notes_path}")
    console.print(f"[green]Expanded summary[/green] → {summary_path}")
    return notes_path, summary_path, chunk_count


def _extract_topic_section(summary_text: str, topic: str) -> str:
    lines = summary_text.splitlines()
    capture = False
    section: list[str] = []
    for line in lines:
        if line.strip() == topic:
            capture = True
            section.append(line)
            continue
        if capture and line and not line.startswith(" ") and line == line.upper()[:1] + line[1:]:
            if line.endswith("Statistics") or "&" in line or "Exam Skills" in line:
                if line != topic:
                    break
        if capture:
            section.append(line)
    return "\n".join(section)
