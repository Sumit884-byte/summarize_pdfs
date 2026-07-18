from summarize_pdfs.pipeline.polish_notes import (
    _extract_polished_text,
    render_polished_notes,
    split_study_notes_by_topic,
)


SAMPLE_NOTES = """\
STATISTICS FOR DATA SCIENCE — QUICK STUDY NOTES
================================================

Expanded with textbook: OpenStax (100 chunks indexed).

• Mean = Σ(xi × f(x)) / n
  where Mean = average; n = count
• Def: mean: The arithmetic average.

PROBABILITY & CONDITIONAL PROBABILITY — Quick Notes

• Def: P(D|B): conditional probability of D given B.
• P(B) = 0.40 [p.217]
  where P(B) = probability of event B
• Def: P(B): The probability of event B occurring.

COMBINATORICS & COUNTING — Quick Notes

• Trick: group constraint → treat block as one unit, then permute
"""


def test_split_study_notes_detects_orphan_descriptive_section():
    header, sections = split_study_notes_by_topic(SAMPLE_NOTES)
    assert "STATISTICS FOR DATA SCIENCE" in header
    assert "Descriptive Statistics" in sections
    assert "Mean = Σ" in sections["Descriptive Statistics"]
    assert "Probability & Conditional Probability" in sections
    assert "P(D|B)" in sections["Probability & Conditional Probability"]
    assert "Combinatorics & Counting" in sections


def test_extract_polished_text_handles_escaped_key():
    data = {"polished\\_text": "PROBABILITY — Quick Notes\n\n• P(A|B)"}
    assert "P(A|B)" in _extract_polished_text(data)


def test_render_polished_notes_preserves_topic_order():
    header, sections = split_study_notes_by_topic(SAMPLE_NOTES)
    polished = {
        "Descriptive Statistics": "DESCRIPTIVE STATISTICS — Quick Notes\n\n• polished mean",
        "Probability & Conditional Probability": "PROBABILITY — Quick Notes\n\n• polished prob",
    }
    text = render_polished_notes(header, polished)
    assert text.index("DESCRIPTIVE STATISTICS") < text.index("PROBABILITY")
    assert header.splitlines()[0] in text
