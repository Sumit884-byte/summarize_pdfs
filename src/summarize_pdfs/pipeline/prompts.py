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
- Write fluid, readable study notes — like a concise textbook section, not a dump of fragments.
- Organize by statistical topic; connect related ideas in natural prose where a bullet can carry a full thought.
- Do NOT use headers like "QUESTION N", "CONCEPTS", "SOURCE EVIDENCE", or numbered mechanical steps.
- Skip hall-ticket confirmations, section metadata, and placeholder instructions.
- Draw content from provided excerpts; mark anything not in excerpts as (standard).
- Prefer one clear statement over three near-duplicates saying the same thing.
""".strip()

DEDUPLICATION_RULES = """
DEDUPLICATION AND MERGING (mandatory):
- ONE definition per concept. If mean, median, or any term appears multiple times with slightly different wording, merge into a single best definition.
- Do NOT repeat the same formula in different wording — keep the clearest version with a complete where_clause.
- Combine overlapping facts into one bullet when they express the same idea (e.g. "mean is average" + "mean = sum/n" → one bullet with definition AND formula).
- Remove near-duplicates before output; err on fewer, richer bullets rather than many redundant ones.
- Def: lines must not restate what a formula bullet already says — pick definition OR formula+where, not both unless they add distinct information.
""".strip()

FORMULA_RULES = """
FORMULAS (mandatory for every formula):
- Copy expressions verbatim from source excerpts when available.
- Every formula MUST include a where_clause defining ALL variables.
  Example: "IQR = interquartile range; Q1 = 25th percentile; Q3 = 75th percentile"
- Use semicolons to separate variable definitions in where_clause.
""".strip()

FORMULA_AWARENESS = """
FORMULA AWARENESS:
- When a concept has a standard formula, pair the definition with that formula (once, not repeatedly).
- Trigger concepts: percentile, quartile, median, mean, mode, range, variance, standard deviation,
  correlation, probability, conditional probability, combination, permutation, Bayes, z-score, IQR.
- When discussing linear transforms (adding b or multiplying by c), state effects on mean, median, AND spread together in connected prose.
""".strip()

DEFINITION_RULES = """
DEFINITIONS:
- Write complete full-sentence definitions — NEVER truncate with "..." or ellipses.
- Each definition must stand alone; merge duplicate defs for the same term into one authoritative sentence.
- When a term has conditions (e.g. mean only for quantitative data, not categorical), include those conditions in the same definition.
""".strip()

FACTS_RULES = """
BRANCH FACTS (from excerpts — do not invent):
- facts[] holds short property/relationship statements for the topic — NOT formulas, NOT term definitions.
- Read the textbook excerpts carefully and extract facts the text actually states or clearly implies.
- Look for: scope conditions (when a measure applies or does not), relationships between concepts,
  branches of the field (descriptive vs inferential), behavior under data transforms, exam-relevant contrasts.
- Write each fact as a complete, fluid sentence — not a telegraphic fragment.
- Do NOT put formulas in facts[] — use formulas[] for expressions with = signs.
""".strip()

CONCEPTUAL_COVERAGE = """
CONCEPTUAL COVERAGE (extract from excerpts when the topic touches these areas — do not skip if the text supports it):
- Branches of statistics: descriptive (summarize observed data) vs inferential (conclude about populations from samples).
- Applicability: which measures of center/spread apply to which data types (e.g. mean requires numeric data; categorical data uses mode).
- Transforms: how adding a constant or multiplying all values affects center (mean, median) and spread (SD, variance).
Only include points backed by the provided excerpts or clearly standard textbook knowledge; mark (standard) if not quoted.
""".strip()

FULL_SUMMARY_RULES = """
FULL SUMMARY MODE (study_guide_complete.txt):
- Topic-organized prose with definitions, formulas (verbatim), and conceptual reasoning.
- NO worked step-by-step calculations with numbers plugged in.
- Explain WHY and HOW concepts apply, not numeric walkthroughs.
- Merge duplicate definitions; write flowing paragraphs in summary_paragraph.
""".strip()

QUICK_NOTES_RULES = """
QUICK NOTES MODE (study_notes.txt):
- Formulas with where_clauses, merged one-line definitions, and exam tricks only.
- NO worked examples, NO numeric calculations, NO multi-step numeric solutions.
- Fewer, denser bullets — each bullet should carry maximum information without repetition.
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
        DEDUPLICATION_RULES,
        FORMULA_RULES,
        FORMULA_AWARENESS,
        DEFINITION_RULES,
        FACTS_RULES,
        CONCEPTUAL_COVERAGE,
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
    "If a question requires combining multiple distinct concepts, explicitly group them as "
    "Co-occurring Concepts in co_occurring_groups. "
    "Always respond with valid JSON only.\n\n"
    + PLAINTEXT_OUTPUT_RULES
    + "\n\n"
    + VOICE_AND_STRUCTURE_RULES
    + "\n\n"
    + DEDUPLICATION_RULES
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
    "Merge duplicates, write fluid prose, and deduplicate before returning JSON. "
    "Always respond with valid JSON only.\n\n"
    + QUALITY_STANDARDS
)

POLISH_NOTES_SYSTEM = (
    "You are an expert statistics exam-prep editor. "
    "Your job is to deduplicate, merge, and polish study notes into fluid readable reference material. "
    "Always respond with valid JSON only.\n\n"
    + PLAINTEXT_OUTPUT_RULES
    + "\n\n"
    + DEDUPLICATION_RULES
    + "\n\n"
    + FORMULA_RULES
    + "\n\n"
    + DEFINITION_RULES
    + "\n\n"
    + FACTS_RULES
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
  "co_occurring_groups": [
    ["Concept A", "Concept B"],
    ["Concept C", "Concept D", "Concept E"]
  ],
  "search_queries": ["3-5 short retrieval queries for OpenStax/textbook lookup"]
}}

Rules:
- Focus on statistical concepts, not exam metadata or administrative steps.
- If the question requires combining multiple distinct concepts, group them in co_occurring_groups.
- Each co_occurring_groups entry must list 2+ concept names that must be used together for this question.
- Use concept names that match the concepts[].name labels where possible.
- One definition per concept — no duplicate labels for the same idea.
- Set has_formula: true for every concept with a standard formula (percentile, median, variance, etc.).
- When has_formula is true, formulas MUST be non-empty with the standard expression and implied variables.
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
  "facts": [
    "fluid factual sentence drawn from excerpts — properties, conditions, relationships (NOT formulas)"
  ],
  "definitions": [
    {{
      "name": "term",
      "text": "single merged full-sentence definition from excerpts",
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
- Extract facts[] from excerpts; include scope, branches, and transform behavior when the text supports it.
- Merge duplicate definitions for the same term into one entry in definitions[].
- Do NOT include worked step-by-step calculations with specific numbers.
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

Existing exam-derived notes (deduplicate and fix errors using textbook):
{existing_notes or "(none yet)"}

Textbook excerpts:
{_format_hits(hits)}

Return JSON:
{{
  "facts": [
    {{"text": "fluid factual sentence from excerpts (not a formula or definition)", "page": 123}}
  ],
  "definitions": [
    {{"name": "term", "text": "single merged full-sentence definition", "page": 123}}
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
  "summary_paragraph": "2-4 fluent sentences weaving the topic together — cite pages like (p. 45); no repetition of bullet content"
}}

Rules:
{QUALITY_STANDARDS}
- Read excerpts thoroughly; extract facts the textbook states about scope, branches, data types, and transforms.
- Merge any duplicate definitions in existing notes + new material into ONE definition per term.
- Do NOT repeat the same idea in facts[], definitions[], and summary_paragraph — each layer adds new value or merge them.
- Use textbook excerpts as authoritative; fix wrong formulas in existing notes.
- Always include page numbers from excerpts.
- If existing notes repeat mean/median/mode definitions, consolidate into one clear definition each.
"""


def polish_notes_prompt(chunk_text: str, topic_name: str) -> str:
    return f"""Polish this topic section of quick study notes for a statistics exam.

Topic: {topic_name}

Input section (likely contains duplicates, repetitive definitions, and awkward phrasing):
{chunk_text}

Return JSON:
{{
  "polished_text": "full polished section text including the topic header line"
}}

Polish rules — DEDUPLICATE AND MERGE FIRST:
- If the same concept (mean, median, mode, probability, etc.) has multiple Def: lines or fact bullets saying the same thing, keep ONE best version and delete the rest.
- Merge overlapping facts into a single richer bullet (e.g. combine "mean is sensitive to outliers" with "median is robust" into one connected sentence when they contrast).
- Do NOT list the same formula twice in different notation — keep the clearest with a complete where-clause.
- Remove Def: lines that merely restate a formula already in the Formulas: section unless the definition adds conditions (e.g. when mean applies vs does not).

Polish rules — FLUID WRITING:
- Each Key Facts bullet should read as a complete, natural sentence — not a telegraphic fragment.
- Connect related ideas: branches of statistics, data-type applicability, transform effects on center and spread.
- Write like a concise study guide a student would actually want to read, not a raw extraction dump.

Polish rules — STRUCTURE:
- Start with: "{topic_name.split(' (')[0].upper()} — Quick Notes" then a blank line.
- "Key Facts:" — 3-6 deduplicated bullets (fewer is fine if merged well).
- Blank line, then "Formulas:" — deduplicated formula bullets with where-clauses on the next line (indented two spaces, starting with "where ").
- Then any remaining Def:/Trick:/Fix: bullets that add information not already covered above.
- Keep textbook page citations like [p.217] when present.
- NO worked examples, NO numeric step-by-step calculations.
- Output plain text inside polished_text — use × for multiplication, no LaTeX or escapes.
- Preserve every DISTINCT fact, formula, and trick — but merged, not repeated.
{DEDUPLICATION_RULES}
{CONCEPTUAL_COVERAGE}
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
