from summarize_pdfs.export.notes import _NotesBucket, _render_notes_topic
from summarize_pdfs.export.summary import _TopicBucket, _render_topic
from summarize_pdfs.export.topic_facts import CANONICAL_TOPIC_FACTS, canonical_facts_for_topic
from summarize_pdfs.models import StudyAnswer
from summarize_pdfs.export.summary import _ingest_answer


def test_canonical_facts_cover_all_topics():
    assert len(CANONICAL_TOPIC_FACTS) >= 7
    for topic in (
        "Descriptive Statistics",
        "Probability & Conditional Probability",
        "Combinatorics & Counting",
    ):
        facts = canonical_facts_for_topic(topic)
        assert len(facts) >= 3
        assert all(len(f) > 10 for f in facts)


def test_notes_render_key_facts_section():
    bucket = _NotesBucket("Descriptive Statistics")
    bucket.add_definition("mean", "arithmetic average of values")
    text = "\n".join(_render_notes_topic(bucket))
    assert "Key Facts:" in text
    assert "Formulas:" in text
    assert "outliers" in text.lower() or "spread" in text.lower()
    assert "Def:" in text


def test_summary_render_key_facts_section():
    bucket = _TopicBucket("Probability & Conditional Probability")
    bucket.add_definition("probability", "measure of likelihood from 0 to 1")
    text = "\n".join(_render_topic(bucket))
    assert "Key facts for this topic:" in text
    assert "0 to 1" in text or "complement" in text.lower()


def test_study_answer_facts_ingested():
    answer = StudyAnswer(
        question_id="q1",
        question_text="Find the mean.",
        concepts=[],
        quotes=[],
        explanation="",
        facts=["Mean is sensitive to outliers; median is robust"],
    )
    bucket = _TopicBucket("Descriptive Statistics")
    _ingest_answer(bucket, answer, None)
    assert any("outliers" in f.lower() for f in bucket.facts.values())


def test_prompts_require_facts_field():
    from summarize_pdfs.pipeline.prompts import (
        expand_topic_user_prompt,
        polish_notes_prompt,
        synthesize_user_prompt,
    )
    from summarize_pdfs.models import ConceptExtraction, ConceptItem, ExamQuestion

    question = ExamQuestion(
        question_id="q1",
        exam_id="e1",
        exam_path="/tmp/exam.pdf",
        number="1",
        text="What is P(A)?",
    )
    concepts = ConceptExtraction(
        question_id="q1",
        concepts=[ConceptItem(name="probability", description="likelihood measure")],
    )
    assert '"facts"' in synthesize_user_prompt(question, concepts, [])
    assert '"facts"' in expand_topic_user_prompt("Probability", "", [], textbook_name="OpenStax")
    assert "Key Facts:" in polish_notes_prompt("• P(A) = n(A)/n(S)", "Probability")
