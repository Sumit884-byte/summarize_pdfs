"""Shared LLM system prompts and quality rules for the study-guide pipeline."""

from __future__ import annotations

from summarize_pdfs.models import ConceptExtraction, ExamQuestion, RetrievedQuote

# ---------------------------------------------------------------------------
# Quality standards (applied across all generation prompts)
# ---------------------------------------------------------------------------

PLAINTEXT_OUTPUT_RULES = """
PLAIN TEXT OUTPUT (mandatory):
- Output plain text only inside JSON string values.
- Use × for multiplication (never \\* or * as escape).
- Use … for ellipsis (never \\... or ...).
- No LaTeX, no markdown, no backslash escapes (\\_, \\(, etc.).
- Write formulas as readable text: √(...), not \\sqrt{...}.
""".strip()

VOICE_AND_STRUCTURE_RULES = """
VOICE AND STRUCTURE:
- Write cohesive study notes organized by statistical topic — NOT per-question Q&A dumps.
- Do NOT use headers like "QUESTION N", "CONCEPTS", "SOURCE EVIDENCE", or numbered mechanical steps.
- Skip hall-ticket confirmations, section metadata, and placeholder instructions.
- Preserve ALL facts, formulas, definitions, and notation from source excerpts — be exhaustive, not terse.
- Prefer textbook/OpenStax definitions from provided excerpts over invented paraphrases.
- Mark content not found in excerpts as (standard) rather than hallucinating.
""".strip()

FORMULA_RULES = """
FORMULAS (mandatory for every formula):
- Copy expressions verbatim from source excerpts when available.
- Every formula MUST include a where_clause defining ALL variables.
  Example: "IQR = interquartile range; Q1 = 25th percentile; Q3 = 75th percentile"
- Use semicolons to separate variable definitions in where_clause.
""".strip()

FORMULA_AWARENESS = """
FORMULA AWARENESS (mandatory):
- If you name ANY statistical concept that has a standard formula, ALWAYS pair it with that formula.
- Never give a bare definition without the formula when a standard formula exists.
- Set has_formula: true on every concept/definition that has a standard formula; include the formula in formulas[].
- Trigger concepts (always include formula when mentioned):
  percentile, quartile, Q1, Q2, Q3, median, mean, mode, range, variance, standard deviation, SD,
  sample variance, sample standard deviation, correlation, Pearson correlation, probability,
  conditional probability, combination, permutation, Bayes, Bayes' theorem, z-score, IQR,
  interquartile range, expected value, complement rule, cumulative frequency.
- Examples:
  percentile → i = (p/100) × n (with where_clause for i, p, n)
  median → middle value rule (odd n: middle; even n: average of two middle values)
  variance → σ² = Σ(xi - μ)² / N
  standard deviation → σ = √(variance)
  correlation → r = Σ[(xi - x̄)(yi - ȳ)] / [(n-1) × sx × sy]
  z-score → z = (x - μ) / σ
  Bayes → P(A|B) = P(B|A) × P(A) / P(B)
  combinations → C(n, k) = n! / (k! × (n - k)!)
  IQR → IQR = Q3 - Q1
""".strip()

DEFINITION_RULES = """
DEFINITIONS:
- Write complete full-sentence definitions — NEVER truncate with "..." or ellipses.
- Each definition must stand alone as a proper sentence ending with a period.
""".strip()

FULL_SUMMARY_RULES = """
FULL SUMMARY MODE (study_guide_complete.txt):
- Topic-organized prose with definitions, formulas (verbatim), and conceptual reasoning.
- NO worked step-by-step calculations with numbers plugged in.
- Explain WHY and HOW concepts apply, not numeric walkthroughs.
""".strip()

QUICK_NOTES_RULES = """
QUICK NOTES MODE (study_notes.txt):
- Formulas with where_clauses, one-line definitions, and exam tricks only.
- NO worked examples, NO numeric calculations, NO multi-step numeric solutions.
""".strip()

BOILERPLATE_SKIP_RULES = """
SKIP BOILERPLATE:
- If the question is a hall-ticket confirmation, section header, or metadata-only entry,
  return {"skip": true} with empty arrays and a brief explanation noting why.
""".strip()

QUALITY_STANDARDS = "\n\n".join(
    [
        PLAINTEXT_OUTPUT_RULES,
        VOICE_AND_STRUCTURE_RULES,
        FORMULA_RULES,
        FORMULA_AWARENESS,
        DEFINITION_RULES,
        FULL_SUMMARY_RULES,
        QUICK_NOTES_RULES,
        BOILERPLATE_SKIP_RULES,
    ]
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_JSON = (
    "You are a precise statistics study-guide assistant. "
    "Always respond with valid JSON only.\n\n"
    + QUALITY_STANDARDS
)

SYSTEM_CONCEPT_EXTRACTION = (
    "You are a statistics exam analyst. "
    "Identify concepts, formulas, and definitions a student must master. "
    "Always respond with valid JSON only.\n\n"
    + PLAINTEXT_OUTPUT_RULES
    + "\n\n"
    + VOICE_AND_STRUCTURE_RULES
    + "\n\n"
    + FORMULA_AWARENESS
)

SYSTEM_EXAM_PARSE = (
    "You are an exam paper parser. Extract clean question text only. "
    "Always respond with valid JSON only.\n\n"
    + BOILERPLATE_SKIP_RULES
    + "\n\nSkip: hall-ticket confirmations, 'ARE YOU SURE YOU HAVE TO WRITE EXAM', "
    "section metadata (Display Question Number, Response Time, etc.), and non-question boilerplate."
)

SYSTEM_EXPAND = (
    "You are expanding exam study notes using textbook excerpts as ground truth. "
    "Always respond with valid JSON only.\n\n"
    + QUALITY_STANDARDS
)

POLISH_NOTES_SYSTEM = (
    "You are an expert statistics exam-prep editor polishing quick-reference study notes. "
    "Always respond with valid JSON only.\n\n"
    + PLAINTEXT_OUTPUT_RULES
    + "\n\n"
    + FORMULA_RULES
    + "\n\n"
    + FORMULA_AWARENESS
    + "\n\n"
    + DEFINITION_RULES
    + "\n\n"
    + QUICK_NOTES_RULES
)


def _format_quotes(quotes: list[RetrievedQuote]) -> str:
    blocks = []
    for i, q in enumerate(quotes, 1):
        blocks.append(f"[{i}] page {q.page} ({q.source_path})\n{q.quote}")
    return "\n\n".join(blocks)


def _format_hits(hits: list[dict]) -> str:
    from pathlib import Path

    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata") or {}
        page = meta.get("page", "?")
        source = Path(meta.get("source_path", "textbook")).name
        blocks.append(f"[{i}] p.{page} ({source})\n{hit['text']}")
    return "\n\n".join(blocks)


def concept_extraction_user_prompt(question: ExamQuestion) -> str:
    return f"""List the statistical concepts, formulas, and definitions required to answer this exam question.

Return JSON:
{{
  "concepts": [
    {{
      "name": "short label",
      "description": "complete one-sentence definition of what to master",
      "has_formula": true,
      "formulas": ["standard formula expression — REQUIRED when has_formula is true"]
    }}
  ],
  "search_queries": ["3-5 short retrieval queries for OpenStax/textbook lookup"]
}}

Rules:
- Focus on statistical concepts, not exam metadata or administrative steps.
- Definitions must be complete sentences (no truncation).
- Set has_formula: true for every concept with a standard formula (percentile, median, variance, etc.).
- When has_formula is true, formulas MUST be non-empty with the standard expression and implied variables.
- Never list a formula-backed concept with an empty formulas array.
{FORMULA_AWARENESS}
{PLAINTEXT_OUTPUT_RULES}

Question ({question.number}, topic={question.topic or "unknown"}):
{question.text}
"""


def synthesize_user_prompt(
    question: ExamQuestion,
    concepts: ConceptExtraction,
    quotes: list[RetrievedQuote],
    *,
    prior_context: str = "",
) -> str:
    concept_lines = "\n".join(
        f"- {c.name}: {c.description}"
        + (f" | formulas: {', '.join(c.formulas)}" if c.formulas else "")
        for c in concepts.concepts
    )
    memory_block = ""
    if prior_context.strip():
        memory_block = (
            "\nPrior study memories (verify against textbook excerpts):\n"
            f"{prior_context.strip()}\n"
        )

    return f"""Build structured study material for this exam question using the textbook excerpts below.

Exam question:
{question.text}

Required concepts:
{concept_lines}
{memory_block}
Source excerpts (copy formulas and definitions VERBATIM; cite page numbers):
{_format_quotes(quotes)}

Return JSON:
{{
  "skip": false,
  "definitions": [
    {{
      "name": "term",
      "text": "complete full-sentence definition from excerpts",
      "has_formula": true
    }}
  ],
  "formulas": [
    {{
      "name": "optional label",
      "expression": "IQR = Q3 - Q1",
      "where_clause": "IQR = interquartile range; Q1 = 25th percentile; Q3 = 75th percentile"
    }}
  ],
  "tricks": ["exam-solving tip without numeric calculations"],
  "reasoning": ["conceptual reasoning — NO numbers plugged in, NO worked arithmetic"],
  "explanation": "2-4 sentences of cohesive prose tying concepts to this question type — NOT a Q&A block",
  "quotes_used": ["exact substring copied from excerpts"]
}}

Rules:
{QUALITY_STANDARDS}
- Do NOT include worked step-by-step calculations with specific numbers.
- Do NOT use Q&A-style headers or per-question dump formatting in explanation.
- List every formula, definition, and fact needed — err on completeness.
- Every definition with has_formula: true MUST have a matching entry in formulas[] with where_clause.
- Prefer quoting source text verbatim over paraphrasing.
"""


def expand_topic_user_prompt(
    topic: str,
    existing_notes: str,
    hits: list[dict],
    *,
    textbook_name: str,
) -> str:
    return f"""Expand exam study notes for one topic using textbook excerpts as ground truth.

Topic: {topic}
Textbook: {textbook_name}

Existing exam-derived notes (fix errors using textbook):
{existing_notes or "(none yet)"}

Textbook excerpts:
{_format_hits(hits)}

Return JSON:
{{
  "definitions": [
    {{"name": "term", "text": "complete full-sentence definition", "page": 123}}
  ],
  "formulas": [
    {{
      "name": "label",
      "expression": "correct formula copied verbatim",
      "where_clause": "var = meaning; var2 = meaning",
      "page": 123
    }}
  ],
  "tricks": [
    {{"text": "exam-solving tip without worked numbers", "page": 123}}
  ],
  "corrections": [
    "Explain errors in existing notes and the textbook-correct version"
  ],
  "summary_paragraph": "2-4 sentence textbook-backed overview citing pages like (p. 45)"
}}

Rules:
{QUALITY_STANDARDS}
- Use textbook excerpts as authoritative; fix wrong formulas.
- Always include page numbers from excerpts.
- Quick-notes fields: formulas + tricks + one-line defs only — no worked examples.
- Full-summary fields: definitions + formulas + conceptual summary_paragraph — no numeric walkthroughs.
- If a definition names a formula-backed concept (percentile, median, variance, etc.), include its formula.
"""


def polish_notes_prompt(chunk_text: str, topic_name: str) -> str:
    return f"""Polish this topic section of quick study notes for a statistics exam.

Topic: {topic_name}

Input section (may contain duplicates, awkward phrasing, or missing cross-links):
{chunk_text}

Return JSON:
{{
  "polished_text": "full polished section text including the topic header line"
}}

Polish rules:
- Merge duplicate formulas and definitions (keep the clearest, most complete version).
- Improve clarity and flow while preserving EVERY fact, formula, trick, and page citation.
- Every formula MUST keep or add a where-clause defining ALL variables on the next line
  (indented with two spaces, starting with "where ").
- If a Def: line names a formula-backed concept (percentile, median, variance, correlation, etc.)
  but no formula bullet exists nearby, ADD the standard formula with where-clause.
- Never leave formula-backed concepts as bare definitions — pair Def: with • formula lines.
{FORMULA_AWARENESS}
- Add brief cross-links where helpful (e.g. P(D|B) relates to Bayes' theorem, conditional probability).
- Compact cheat-sheet style: bullet lines starting with • for formulas, Def:, Trick:, Fix:.
- NO worked examples, NO numeric step-by-step calculations, NO multi-step numeric solutions.
- Keep textbook page citations like [p.217] when present.
- Start the section with: "{topic_name.split(' (')[0].upper()} — Quick Notes" then a blank line.
- Output plain text inside polished_text — use × for multiplication, … for ellipsis, no LaTeX or escapes.
- Do NOT drop content; err on completeness over brevity.
"""


def exam_parse_user_prompt(pages_text: str) -> str:
    return f"""Extract every substantive exam question from this paper as JSON.

Return:
{{
  "questions": [
    {{
      "number": "1",
      "text": "clean question text only — no metadata prefixes",
      "topic": "short statistical topic label or null",
      "official_answer": "marking scheme answer if present, else null",
      "marks": 5
    }}
  ]
}}

Rules:
{BOILERPLATE_SKIP_RULES}
- Strip Display Question Number, Response Time, Calculator, Question Label metadata from text.
- Skip hall-ticket and confirmation questions entirely.
- Keep full question wording for substantive statistics questions.

Exam text:
{pages_text[:120000]}
"""
