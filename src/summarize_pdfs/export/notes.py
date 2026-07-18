from __future__ import annotations

import re
from pathlib import Path

from summarize_pdfs.export.formula_glossary import (
    CANONICAL_TOPIC_FORMULAS,
    canonical_formulas_for_topic,
    formulas_for_concept,
    render_formula_lines,
)
from summarize_pdfs.export.plaintext import sanitize_plaintext
from summarize_pdfs.export.summary import (
    TOPIC_ORDER,
    _FRAGMENT_FORMULA_RE,
    _GENERIC_CONCEPT_RE,
    _dedupe_key,
    _effective_question_text,
    _formula_quality,
    _infer_topic,
    _is_boilerplate,
    _is_junk,
    _is_placeholder,
    _normalize,
    _parse_explanation,
    load_answers_from_json,
    load_questions_from_jsonl,
)
from summarize_pdfs.models import ExamQuestion, StudyAnswer

_WORKED_RE = re.compile(
    r"(?:"
    r"Therefore,|approximately \d+%|=\s*\d+/\d+|=\s*0\.\d+|"
    r"=\s*\d+\.\d+|√\[|\(\d+\s*-\s*[\d.]+\)\^2|"
    r"we need to (?:first|count|find the number|multiply)|"
    r"Enter the answer|P\([A-Z]\)\s*=\s*\d+/|"
    r"There are \d+|given as \d+|n\(S\)\s*=\s*\d+"
    r")",
    re.IGNORECASE,
)

_JUNK_FORMULA_RE = re.compile(
    r"(?:"
    r"^f\(x\)\s*=\s*\d+\.\d+|^X\s*=\s*\d+m|"
    r"^median\s*=\s*\(x1\s*\+|^\(\d+\s*[\+\-\*]|^\(\d+,\d+|"
    r"nCr formula for combinations|^none$|"
    r"Selection of (?:people|boys|performers|leaders)|"
    r"will occur simultaneously|will happen with replacement|"
    r"^Pearson Correlation Coefficient$"
    r")",
    re.IGNORECASE,
)

_INCOMPLETE_RE = re.compile(
    r"(?:"
    r"^To find|^To determine|^To solve|^So, the|^where P\(B\)|"
    r"first calculate the first and$|"
    r"The probability of a person having ALS is 1%|"
    r"Similarly, the individual frequency|"
    r"The formula for finding the number"
    r")",
    re.IGNORECASE,
)

_VALID_FORMULA_RE = re.compile(
    r"(?:"
    r"P\s*\(|Var\s*\(|SD\s*\(|σ|μ|n!|C\s*\(|"
    r"IQR|Q[123]|median|mean|frequency|correlation|percentile|quartile|"
    r"=\s*[a-zA-Z(]|/\s*n\b|\^2|choose"
    r")",
    re.IGNORECASE,
)

_VAGUE_DEF_RE = re.compile(
    r"^(?:Calculate|Understanding of|Knowledge of|knowing how|Learn about)",
    re.IGNORECASE,
)

_TIP_KEYWORD_RE = re.compile(
    r"(?:"
    r"at least|complement|given that|conditional|independen|replacement|"
    r"order matters|permutation|combination|circular|Bayes|without replacement|"
    r"simultaneously|cross sectional|time series|IQR|outlier|quartile|"
    r"mean.*median|Var\(|standard deviation|correlation|contingency|"
    r"nominal|ordinal|interval|ratio"
    r")",
    re.IGNORECASE,
)

_QUESTION_PATTERN_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "Trick",
        re.compile(r"at least (?:one|a)", re.I),
        '"at least one" → use complement 1 - P(none)',
    ),
    (
        "Trick",
        re.compile(r"given that|conditional", re.I),
        '"given that" → conditional probability P(A|B), check Bayes',
    ),
    (
        "Trick",
        re.compile(r"same colour|both are boys|co-education.*principal", re.I),
        "Conditional probability — restrict sample space first",
    ),
    (
        "MSQ pattern",
        re.compile(r"Multiple Select|MSQ|Selectable Option", re.I),
        "check independence vs replacement; order matters vs combination",
    ),
    (
        "MCQ pattern",
        re.compile(r"Multiple Choice|MCQ", re.I),
        "eliminate by scale/type rules before calculating",
    ),
    (
        "SA pattern",
        re.compile(r"Short Answer|Question Type : SA", re.I),
        "numeric answer — show formula then compute; watch decimal accuracy",
    ),
    (
        "Trick",
        re.compile(r"vowels always come together|sit together|circular", re.I),
        "group constraint → treat block as one unit, then permute",
    ),
    (
        "Trick",
        re.compile(r"subtracted.*multiplied|modified marks|linear transform", re.I),
        "linear transform: new mean = c·old + b; new SD = |c|·old SD",
    ),
]

_MAX_DEF_LEN = 300
_MAX_TIP_LEN = 120
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_JUNK_TIP_WORDS = frozenset({"none", "n/a", "na"})


def _is_valid_formula(formula: str) -> bool:
    formula = formula.strip()
    if not formula or formula.lower() in ("none", "n/a") or _is_junk(formula):
        return False
    if _FRAGMENT_FORMULA_RE.match(formula):
        return False
    if _JUNK_FORMULA_RE.search(formula) or _INCOMPLETE_RE.search(formula):
        return False
    if re.match(r"^The total number of ways", formula, re.I):
        return False
    if not _VALID_FORMULA_RE.search(formula):
        return False
    if len(re.findall(r"\b\d+\b", formula)) >= 3:
        return False
    return True


def _is_worked_content(text: str) -> bool:
    text = text.strip()
    if len(text) > _MAX_TIP_LEN:
        return True
    if _WORKED_RE.search(text) or _INCOMPLETE_RE.search(text):
        return True
    if re.match(r"^(?:To find|To determine|To solve|The formula for|So, the|The total number of ways)", text, re.I):
        return True
    if len(re.findall(r"\d+", text)) >= 4:
        return True
    if re.search(r"\d+%", text):
        return True
    return False


def _fit_definition(text: str, max_len: int = _MAX_DEF_LEN) -> str:
    """Keep full definition text, or up to 3 sentences within max_len — no ellipsis truncation."""
    text = re.sub(r"\s+", " ", text.strip()).rstrip(".")
    if len(text) <= max_len:
        return text

    sentences = _SENTENCE_END_RE.split(text)
    if len(sentences) > 1:
        kept: list[str] = []
        for sentence in sentences:
            candidate = " ".join(kept + [sentence]).strip()
            if len(candidate) > max_len or len(kept) >= 3:
                break
            kept.append(sentence)
        if kept:
            return " ".join(kept).rstrip(".")

    chunk = text[:max_len]
    last_period = max(chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "))
    if last_period >= max_len // 3:
        return text[: last_period + 1].strip().rstrip(".")

    return text[:max_len].rsplit(" ", 1)[0]


def _is_tip_candidate(text: str) -> bool:
    text = text.strip()
    if not text or text.lower() in _JUNK_TIP_WORDS or _is_junk(text) or _is_worked_content(text):
        return False
    if _TIP_KEYWORD_RE.search(text):
        return True
    if text.lower().startswith(("remember", "note:", "trick:", "when ", "if ")):
        return True
    return 15 <= len(text) <= 80 and "=" not in text


def _formula_to_trick(formula: str) -> str | None:
    """Turn complement/at-least style formulas into short tricks."""
    lower = formula.lower()
    if "at least" in lower and "1 -" in lower.replace(" ", ""):
        return '"at least one" → use complement 1 - P(none)'
    if "not a" in lower or "p(not" in lower.replace(" ", ""):
        return "complement rule: P(not A) = 1 - P(A)"
    return None


class _NotesBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.formulas: dict[str, str] = {}
        self.definitions: dict[str, str] = {}
        self.tips: dict[str, str] = {}

    def add_formula(self, formula: str) -> None:
        formula = sanitize_plaintext(formula.strip())
        if not _is_valid_formula(formula):
            return
        norm = _normalize(formula.replace("×", "*"))
        norm_key = re.sub(r"\s+", "", norm)[:80]
        existing = self.formulas.get(norm_key)
        if existing is None or _formula_quality(formula) > _formula_quality(existing):
            self.formulas[norm_key] = formula
        trick = _formula_to_trick(formula)
        if trick:
            key = _dedupe_key(trick)
            if not trick.lower().startswith(("trick:", "mcq", "msq", "sa ")):
                trick = f"Trick: {trick}"
            self.tips[key] = trick

    def add_definition(self, name: str, description: str) -> None:
        if _GENERIC_CONCEPT_RE.match(name.strip()) or _is_junk(name):
            return
        if _is_placeholder(description) and _is_placeholder(name):
            return
        desc = description.strip()
        if not desc or _is_worked_content(desc) or _VAGUE_DEF_RE.match(desc):
            return
        if re.search(r"\bunderstand(?:ing)?\b", desc, re.I):
            return
        line = desc if desc.lower().startswith(name.lower()) else f"{name}: {desc}"
        line = sanitize_plaintext(_fit_definition(line))
        self.definitions[_dedupe_key(name)] = line
        for formula in formulas_for_concept(name):
            self.add_formula(formula)

    def add_tip(self, tip: str, *, prefix: str = "Trick") -> None:
        tip = sanitize_plaintext(tip.strip())
        if not tip or _is_junk(tip) or _is_worked_content(tip):
            return
        if _is_valid_formula(tip):
            self.add_formula(tip)
            return
        if not tip.lower().startswith(("trick:", "mcq", "msq", "sa ")):
            tip = f"{prefix}: {tip}" if prefix else tip
        self.tips[_dedupe_key(tip)] = tip

    def is_empty(self) -> bool:
        return not (self.formulas or self.definitions or self.tips)


def _extract_question_patterns(
    answer: StudyAnswer,
    question: ExamQuestion | None,
) -> list[str]:
    blob = _effective_question_text(answer, question)
    if len(blob) < 25:
        return []
    patterns: list[str] = []
    seen: set[str] = set()
    for prefix, pattern, hint in _QUESTION_PATTERN_RULES:
        if not pattern.search(blob):
            continue
        key = _dedupe_key(hint)
        if key in seen:
            continue
        seen.add(key)
        patterns.append(f"{prefix}: {hint}")
    return patterns


def _formula_line(formula) -> str:
    if isinstance(formula, str):
        return sanitize_plaintext(formula.strip())
    name = sanitize_plaintext(getattr(formula, "name", "") or "")
    expr = sanitize_plaintext(getattr(formula, "expression", "") or "")
    where = sanitize_plaintext(getattr(formula, "where_clause", "") or "")
    if name and expr and "=" not in expr:
        line = f"{name}: {expr}"
    else:
        line = expr
    if where and "where" not in line.lower():
        line = f"{line}, where {where}"
    return line


def _ingest_notes(
    bucket: _NotesBucket,
    answer: StudyAnswer,
    question: ExamQuestion | None,
) -> None:
    for defn in answer.definitions:
        bucket.add_definition(defn.name, defn.text)

    for formula in answer.formulas:
        bucket.add_formula(_formula_line(formula))

    for concept in answer.concepts:
        bucket.add_definition(concept.name, concept.description)
        for formula in concept.formulas:
            bucket.add_formula(formula)

    extra_formulas, facts, _prose = _parse_explanation(answer.explanation)
    for formula in extra_formulas:
        bucket.add_formula(formula)
    for fact in facts:
        if _is_tip_candidate(fact):
            bucket.add_tip(fact)

    for trick in answer.tricks:
        bucket.add_tip(trick)

    for step in answer.steps:
        if _is_tip_candidate(step):
            bucket.add_tip(step, prefix="")

    for pattern in _extract_question_patterns(answer, question):
        bucket.tips[_dedupe_key(pattern)] = pattern


def _short_topic_label(topic: str) -> str:
    return topic.split(" (")[0].upper() if " (" in topic else topic.upper()


def _concept_from_definition(defn: str) -> str:
    """Extract concept label from a Def: line for glossary lookup."""
    text = re.sub(r"^def:\s*", "", defn.strip(), flags=re.I)
    if ":" in text:
        return text.split(":", 1)[0].strip()
    return text.split()[0] if text else ""


def _render_notes_topic(bucket: _NotesBucket) -> list[str]:
    label = _short_topic_label(bucket.name)
    lines: list[str] = [f"{label} — Quick Notes", ""]

    seen_norm: set[str] = set()
    for formula in canonical_formulas_for_topic(bucket.name):
        lhs = formula.split("=")[0] if "=" in formula else formula
        norm = re.sub(r"\s+", "", _normalize(lhs.replace("×", "*")))[:30]
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        lines.extend(render_formula_lines(formula))

    for formula in sorted(
        bucket.formulas.values(),
        key=lambda f: (-_formula_quality(f), f.lower()),
    ):
        lhs = formula.split("=")[0] if "=" in formula else formula
        norm = re.sub(r"\s+", "", _normalize(lhs.replace("×", "*")))[:30]
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        lines.extend(render_formula_lines(formula))

    for definition in sorted(bucket.definitions.values(), key=str.lower):
        if not definition.lower().startswith("def:"):
            definition = f"Def: {definition}"
        lines.append(f"• {definition}")
        concept = _concept_from_definition(definition)
        for formula in formulas_for_concept(concept):
            norm = re.sub(r"\s+", "", _normalize(formula.replace("×", "*")))[:80]
            if norm not in seen_norm:
                seen_norm.add(norm)
                lines.extend(render_formula_lines(formula))

    for tip in sorted(bucket.tips.values(), key=str.lower):
        lines.append(f"• {tip}")

    lines.append("")
    return lines


def render_notes_txt(
    answers: list[StudyAnswer],
    questions: list[ExamQuestion] | None = None,
) -> str:
    """Render compact cheat-sheet notes: formulas, defs, tricks — no worked problems."""
    by_id = {q.question_id: q for q in (questions or [])}
    buckets: dict[str, _NotesBucket] = {}

    kept = 0
    skipped = 0
    for answer in answers:
        if _is_boilerplate(answer):
            skipped += 1
            continue
        kept += 1
        topic = _infer_topic(answer, by_id.get(answer.question_id))
        if topic not in buckets:
            buckets[topic] = _NotesBucket(topic)
        _ingest_notes(buckets[topic], answer, by_id.get(answer.question_id))

    lines = [
        "STATISTICS FOR DATA SCIENCE — QUICK STUDY NOTES",
        "=" * 48,
        "",
        "Compact reference: formulas, one-line definitions, tricks & question patterns.",
        f"From {kept} substantive entries ({skipped} boilerplate omitted). No worked examples.",
        "",
    ]

    for topic in TOPIC_ORDER:
        if topic in CANONICAL_TOPIC_FORMULAS and topic not in buckets:
            buckets[topic] = _NotesBucket(topic)

    ordered_topics = [
        t
        for t in TOPIC_ORDER
        if t in buckets
        and (canonical_formulas_for_topic(t) or not buckets[t].is_empty())
    ]
    for topic in ordered_topics:
        lines.extend(_render_notes_topic(buckets[topic]))

    if not ordered_topics:
        lines.append("No substantive content found after filtering boilerplate entries.")

    return "\n".join(lines).rstrip() + "\n"


def export_notes_file(
    json_path: Path,
    output_path: Path,
    questions_path: Path | None = None,
) -> Path:
    answers = load_answers_from_json(json_path)
    questions = load_questions_from_jsonl(questions_path) if questions_path else None
    text = render_notes_txt(answers, questions)
    output_path.write_text(text)
    return output_path
