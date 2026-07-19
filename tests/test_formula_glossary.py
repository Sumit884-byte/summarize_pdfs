from summarize_pdfs.export.formula_glossary import (
    annotate_formula_parts,
    extract_formula_variables,
    render_formula_lines,
)


def test_iqr_variables():
    vars_ = extract_formula_variables("IQR = Q3 - Q1")
    assert "IQR" in vars_
    assert "Q3" in vars_
    assert "Q1" in vars_


def test_iqr_glossary():
    display, where = annotate_formula_parts("IQR = Q3 - Q1")
    assert display == "IQR = Q3 - Q1"
    assert where is not None
    assert "IQR = interquartile range" in where
    assert "Q1 = first quartile" in where
    assert "Q3 = third quartile" in where


def test_bayes_glossary():
    display, where = annotate_formula_parts("P(A|B) = P(B|A) × P(A) / P(B)")
    assert "P(A|B)" in display
    assert where is not None
    assert "P(A|B)" in where
    assert "P(B|A)" in where
    assert "P(A)" in where
    assert "P(B)" in where


def test_preserves_existing_where():
    formula = (
        "P(A) = n(A)/n(S), where P(A) is the probability of event A occurring, "
        "n(A) is the number of outcomes in event A, and n(S) is the total number of possible outcomes."
    )
    display, where = annotate_formula_parts(formula)
    assert display == "P(A) = n(A)/n(S)"
    assert where is not None
    assert "probability of event A" in where


def test_render_multiline():
    lines = render_formula_lines("IQR = Q3 - Q1")
    assert len(lines) == 2
    assert lines[0].startswith("• IQR = Q3 - Q1")
    assert lines[1].startswith("  where ")


def test_percentile_position_glossary():
    display, where = annotate_formula_parts("Percentile position: i = (p/100) × n")
    assert display == "Percentile position: i = (p/100) × n"
    assert where is not None
    assert "i = index/position" in where
    assert "p = percentile" in where
    assert "n = " in where


def test_quartile_percentile_relationship():
    display, where = annotate_formula_parts(
        "Quartiles: Q1 = 25th percentile; Q2 = median = 50th percentile; Q3 = 75th percentile"
    )
    assert "Q1" in display
    assert where is not None
    assert "Q1 = first quartile" in where
    assert "Q2 = second quartile" in where
    assert "Q3 = third quartile" in where


def test_canonical_formulas_for_topic():
    from summarize_pdfs.export.formula_glossary import canonical_formulas_for_topic

    formulas = canonical_formulas_for_topic("Frequency & Distribution")
    assert any("Percentile position" in f for f in formulas)
    assert any("Quartiles:" in f for f in formulas)
    assert any("interpolation" in f for f in formulas)


def test_combinations_glossary():
    display, where = annotate_formula_parts("C(n, k) = n! / (k! × (n - k)!)")
    assert where is not None
    assert "C(n, k)" in where
    assert "n!" in where or "n = " in where


def test_concept_to_formula_lookup():
    from summarize_pdfs.export.formula_glossary import (
        concept_has_formula,
        formula_for_concept,
        formulas_for_concept,
    )

    assert concept_has_formula("percentile")
    assert "i = (p/100)" in formula_for_concept("percentile")
    assert formula_for_concept("Standard Deviation") is not None
    assert formula_for_concept("z-score") is not None
    assert formulas_for_concept("median")


def test_notes_injects_formula_for_percentile_definition():
    from summarize_pdfs.export.notes import _NotesBucket, _render_notes_topic

    bucket = _NotesBucket("Descriptive Statistics")
    bucket.add_definition("percentile", "value below which a given percentage of data falls")
    text = "\n".join(_render_notes_topic(bucket))
    assert "Formulas:" in text
    assert "percentile" in text.lower()
    assert "i = (p/100)" in text or "Percentile position" in text
    assert "where " in text
