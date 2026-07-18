from __future__ import annotations

from summarize_pdfs.config import AppConfig
from summarize_pdfs.export.formula_glossary import formulas_for_concept
from summarize_pdfs.models import ConceptExtraction, ConceptItem, ExamQuestion
from summarize_pdfs.pipeline.llm import LLMClient, chat_json
from summarize_pdfs.pipeline.prompts import SYSTEM_CONCEPT_EXTRACTION, concept_extraction_user_prompt


def _normalize_concept_payload(data: object) -> dict:
    """Accept dict or mixed list payloads from LLM JSON."""
    if isinstance(data, dict):
        return _sanitize_concept_dict(data)

    if isinstance(data, list):
        concepts: list[dict] = []
        search_queries: list[str] = []
        co_occurring_groups: list = []

        for item in data:
            if isinstance(item, dict):
                if "concepts" in item:
                    inner = _sanitize_concept_dict(item)
                    concepts.extend(inner.get("concepts") or [])
                    search_queries.extend(inner.get("search_queries") or [])
                    co_occurring_groups.extend(inner.get("co_occurring_groups") or [])
                elif _concept_field(item, "name"):
                    concepts.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str) and sub.strip():
                        search_queries.append(sub.strip())
            elif isinstance(item, str) and item.strip():
                search_queries.append(item.strip())

        return {
            "concepts": concepts,
            "search_queries": search_queries,
            "co_occurring_groups": co_occurring_groups,
        }

    return {}


def _concept_field(item: dict, key: str):
    """Read concept fields; LLM sometimes escapes underscores in keys."""
    if key in item:
        return item[key]
    escaped = key.replace("_", "\\_")
    return item.get(escaped)


def _sanitize_concept_dict(data: dict) -> dict:
    concepts = [c for c in (data.get("concepts") or []) if isinstance(c, dict)]
    co_groups = data.get("co_occurring_groups")
    if co_groups is None:
        co_groups = data.get("co\\_occurring\\_groups") or []
    return {
        "concepts": concepts,
        "search_queries": list(data.get("search_queries") or []),
        "co_occurring_groups": co_groups,
    }


async def extract_concepts(
    question: ExamQuestion,
    *,
    client: LLMClient,
    config: AppConfig,
) -> ConceptExtraction:
    data = await chat_json(
        client,
        model=config.llm.model,
        prompt=concept_extraction_user_prompt(question),
        temperature=config.llm.temperature,
        cache_dir=config.cache_dir,
        system_prompt=SYSTEM_CONCEPT_EXTRACTION,
    )
    data = _normalize_concept_payload(data)
    concepts = []
    for c in data.get("concepts", []):
        if not isinstance(c, dict):
            continue
        name = _concept_field(c, "name") or "concept"
        formulas = list(_concept_field(c, "formulas") or [])
        has_formula = bool(_concept_field(c, "has_formula"))
        if not formulas and (has_formula or formulas_for_concept(name)):
            formulas = formulas_for_concept(name)
            has_formula = bool(formulas)
        concepts.append(
            ConceptItem(
                name=name,
                description=c.get("description", ""),
                formulas=formulas,
                has_formula=has_formula or bool(formulas),
            )
        )
    return ConceptExtraction(
        question_id=question.question_id,
        concepts=concepts,
        search_queries=data.get("search_queries") or [],
        co_occurring_groups=_parse_co_occurring_groups(data.get("co_occurring_groups")),
    )


def _parse_co_occurring_groups(raw) -> list[list[str]]:
    if not raw:
        return []
    groups: list[list[str]] = []
    for group in raw:
        if not isinstance(group, list):
            continue
        names = [str(name).strip() for name in group if str(name).strip()]
        if len(names) >= 2:
            groups.append(names)
    return groups
