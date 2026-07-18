from __future__ import annotations

from summarize_pdfs.config import AppConfig
from summarize_pdfs.export.formula_glossary import formulas_for_concept
from summarize_pdfs.models import ConceptExtraction, ConceptItem, ExamQuestion
from summarize_pdfs.pipeline.llm import LLMClient, chat_json
from summarize_pdfs.pipeline.prompts import SYSTEM_CONCEPT_EXTRACTION, concept_extraction_user_prompt


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
    concepts = []
    for c in data.get("concepts", []):
        name = c.get("name", "concept")
        formulas = list(c.get("formulas") or [])
        has_formula = bool(c.get("has_formula"))
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
