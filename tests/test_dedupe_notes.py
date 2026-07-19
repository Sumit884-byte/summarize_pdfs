from summarize_pdfs.export.dedupe_notes import deduplicate_study_notes

SAMPLE_WITH_SUBSECTIONS = """\
STATISTICS — QUICK STUDY NOTES
===============================

EXAM CONCEPT CLUSTERS (from co-occurrence analysis)
----------------------------------------------------
• Percentile + Quartile (appears together in 11 questions)

DESCRIPTIVE STATISTICS — Quick Notes

Key Facts:
• Mean is sensitive to outliers; median is robust to them
• Mean is sensitive to outliers; median is robust to them

Formulas:
• IQR = Q3 - Q1
  where Q3 = third quartile; Q1 = first quartile
• IQR = Q3 - Q1
  where Q3 = third quartile; Q1 = first quartile
• mean = sum(x) / n
  where n = count

PROBABILITY & CONDITIONAL PROBABILITY — Quick Notes

Key Facts:
• Conditional probability updates beliefs with new evidence

Formulas:
• IQR = Q3 - Q1
  where Q3 = third quartile; Q1 = first quartile
• P(A|B) = P(A and B) / P(B)
  where P(A|B) = probability of event A given event B has already occurred
"""


def test_dedupe_preserves_key_facts_and_formulas_headers():
    result = deduplicate_study_notes(SAMPLE_WITH_SUBSECTIONS, global_formulas=False)
    assert "Key Facts:" in result
    assert "Formulas:" in result
    assert result.count("Key Facts:") == 2
    assert result.count("Formulas:") == 2


def test_dedupe_deduplicates_within_subsections():
    result = deduplicate_study_notes(SAMPLE_WITH_SUBSECTIONS, global_formulas=False)
    assert result.count("Mean is sensitive to outliers") == 1
    descriptive, probability = result.split("PROBABILITY & CONDITIONAL PROBABILITY — Quick Notes", 1)
    assert descriptive.count("IQR = Q3 - Q1") == 1
    assert probability.count("IQR = Q3 - Q1") == 1


def test_dedupe_global_formula_dedup_across_topics():
    result = deduplicate_study_notes(SAMPLE_WITH_SUBSECTIONS, global_formulas=True)
    assert result.count("IQR = Q3 - Q1") == 1
    assert "P(A|B)" in result
