from __future__ import annotations

from summarize_pdfs.config import AppConfig
from summarize_pdfs.export.plaintext import sanitize_plaintext
from summarize_pdfs.models import (
    ConceptExtraction,
    DefinitionItem,
    ExamQuestion,
    FormulaItem,
    RetrievedQuote,
    StudyAnswer,
)
from summarize_pdfs.pipeline.llm import LLMClient, chat_json, coerce_json_dict
from summarize_pdfs.pipeline.prompts import SYSTEM_JSON, synthesize_user_prompt


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
        return out
    return [str(value)]


def _parse_definitions(raw) -> list[DefinitionItem]:
    items: list[DefinitionItem] = []
    for entry in raw or []:
        if isinstance(entry, str) and entry.strip():
            items.append(DefinitionItem(name="definition", text=sanitize_plaintext(entry)))
        elif isinstance(entry, dict):
            name = sanitize_plaintext(str(entry.get("name") or "definition"))
            text = sanitize_plaintext(str(entry.get("text") or entry.get("description") or ""))
            if text:
                items.append(DefinitionItem(name=name, text=text))
    return items


def _parse_formulas(raw) -> list[FormulaItem]:
    items: list[FormulaItem] = []
    for entry in raw or []:
        if isinstance(entry, str) and entry.strip():
            items.append(FormulaItem(expression=sanitize_plaintext(entry)))
        elif isinstance(entry, dict):
            expr = sanitize_plaintext(str(entry.get("expression") or entry.get("text") or ""))
            if not expr:
                continue
            where = entry.get("where_clause") or entry.get("where")
            items.append(
                FormulaItem(
                    name=sanitize_plaintext(str(entry.get("name") or "")) or None,
                    expression=expr,
                    where_clause=sanitize_plaintext(str(where)) if where else None,
                )
            )
    return items


async def synthesize_answer(
    question: ExamQuestion,
    concepts: ConceptExtraction,
    quotes: list[RetrievedQuote],
    *,
    client: LLMClient,
    config: AppConfig,
    prior_context: str = "",
) -> StudyAnswer:
    data = coerce_json_dict(
        await chat_json(
            client,
            model=config.llm.model,
            prompt=synthesize_user_prompt(
                question,
                concepts,
                quotes,
                prior_context=prior_context,
            ),
            temperature=config.llm.temperature,
            cache_dir=config.cache_dir,
            system_prompt=SYSTEM_JSON,
        )
    )

    if data.get("skip"):
        return StudyAnswer(
            question_id=question.question_id,
            question_text=question.text,
            concepts=concepts.concepts,
            quotes=[],
            explanation=sanitize_plaintext(str(data.get("explanation") or "")),
            skipped=True,
        )

    used = set(data.get("quotes_used") or [])
    filtered_quotes = [q for q in quotes if any(u in q.quote or q.quote in u for u in used)] or quotes

    definitions = _parse_definitions(data.get("definitions"))
    formulas = _parse_formulas(data.get("formulas"))
    facts = [sanitize_plaintext(f) for f in _as_str_list(data.get("facts"))]
    tricks = [sanitize_plaintext(t) for t in _as_str_list(data.get("tricks"))]
    reasoning = [sanitize_plaintext(r) for r in _as_str_list(data.get("reasoning"))]
    explanation = sanitize_plaintext(str(data.get("explanation") or ""))

    # Backward compatibility: legacy LLM responses with key_facts / steps
    if not facts:
        facts = [sanitize_plaintext(f) for f in _as_str_list(data.get("key_facts"))]
    if not definitions and not data.get("facts"):
        definitions = _parse_definitions(data.get("key_facts"))
    legacy_steps = _as_str_list(data.get("steps"))
    if not reasoning and legacy_steps:
        reasoning = [sanitize_plaintext(s) for s in legacy_steps]

    return StudyAnswer(
        question_id=question.question_id,
        question_text=question.text,
        concepts=concepts.concepts,
        quotes=filtered_quotes,
        explanation=explanation,
        steps=legacy_steps,
        definitions=definitions,
        formulas=formulas,
        facts=facts,
        tricks=tricks,
        reasoning=reasoning,
    )
