from summarize_pdfs.pipeline.prompts import (
    PLAINTEXT_OUTPUT_RULES,
    POLISH_NOTES_SYSTEM,
    QUALITY_STANDARDS,
    SYSTEM_CONCEPT_EXTRACTION,
    SYSTEM_JSON,
    concept_extraction_user_prompt,
    exam_parse_user_prompt,
    expand_topic_user_prompt,
    polish_notes_prompt,
    synthesize_user_prompt,
)
from summarize_pdfs.models import ConceptExtraction, ConceptItem, ExamQuestion


def test_quality_standards_include_plaintext_rules():
    assert "×" in PLAINTEXT_OUTPUT_RULES
    assert "where_clause" in QUALITY_STANDARDS or "where" in QUALITY_STANDARDS.lower()
    assert "DEDUPLICATION" in QUALITY_STANDARDS or "duplicate" in QUALITY_STANDARDS.lower()
    assert "fluid" in QUALITY_STANDARDS.lower() or "merge" in QUALITY_STANDARDS.lower()


def test_system_json_includes_quality():
    assert "plain text" in SYSTEM_JSON.lower()
    assert "JSON" in SYSTEM_JSON


def test_synthesize_prompt_structure():
    question = ExamQuestion(
        question_id="q1",
        exam_id="e1",
        exam_path="/tmp/exam.pdf",
        number="1",
        text="What is the mean of 2, 4, 6?",
    )
    concepts = ConceptExtraction(
        question_id="q1",
        concepts=[ConceptItem(name="mean", description="average of values")],
        search_queries=["sample mean formula"],
    )
    prompt = synthesize_user_prompt(question, concepts, [])
    assert "where_clause" in prompt
    assert "has_formula" in prompt
    assert '"facts"' in prompt
    assert "QUESTION N" not in prompt or "NOT" in prompt
    assert "plain text" in prompt.lower()


def test_concept_system_prompt_co_occurrence():
    assert "Co-occurring Concepts" in SYSTEM_CONCEPT_EXTRACTION
    assert "co_occurring_groups" in SYSTEM_CONCEPT_EXTRACTION


def test_concept_prompt_no_mechanical_steps():
    question = ExamQuestion(
        question_id="q1",
        exam_id="e1",
        exam_path="/tmp/exam.pdf",
        number="1",
        text="Find P(A|B).",
    )
    prompt = concept_extraction_user_prompt(question)
    assert "search_queries" in prompt
    assert "co_occurring_groups" in prompt
    assert "has_formula" in prompt
    assert "Find P(A|B)" in prompt


def test_expand_prompt_includes_formula_glossary():
    prompt = expand_topic_user_prompt("Descriptive Statistics", "", [], textbook_name="OpenStax")
    assert "where_clause" in prompt
    assert '"facts"' in prompt
    assert "OpenStax" in prompt or "textbook" in prompt.lower()


def test_exam_parse_skips_boilerplate():
    prompt = exam_parse_user_prompt("ARE YOU SURE YOU HAVE TO WRITE EXAM")
    assert "hall-ticket" in prompt.lower() or "boilerplate" in prompt.lower()


def test_polish_prompt_includes_rules():
    prompt = polish_notes_prompt("• Def: P(D|B): conditional", "Probability & Conditional Probability")
    assert "polished_text" in prompt
    assert "merge" in prompt.lower() or "deduplicate" in prompt.lower()
    assert "where" in prompt.lower()
    assert "Probability & Conditional Probability" in prompt


def test_polish_prompt_merges_duplicate_definitions():
    prompt = polish_notes_prompt(
        "• Def: mean: average\n• Def: mean: the average value\n• Def: Mean: arithmetic mean",
        "Descriptive Statistics",
    )
    assert "ONE" in prompt or "one" in prompt.lower()
    assert "merge" in prompt.lower() or "duplicate" in prompt.lower()


def test_polish_system_includes_quick_notes():
    assert "JSON" in POLISH_NOTES_SYSTEM
    assert "×" in POLISH_NOTES_SYSTEM or "plain text" in POLISH_NOTES_SYSTEM.lower()
