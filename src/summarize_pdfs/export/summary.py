from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from summarize_pdfs.export.formula_glossary import render_formula_lines
from summarize_pdfs.export.plaintext import sanitize_plaintext
from summarize_pdfs.models import ExamQuestion, StudyAnswer

_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"ARE YOU SURE YOU HAVE TO WRITE EXAM|"
    r"HALL TICKET|"
    r"THIS IS QUESTION PAPER FOR THE SUBJECT|"
    r"REGISTERED BY YOU|"
    r"SEMESTER II:\s*(?:ENGLISH|MATHEMATICS|INTRODUCTION TO PYTHON)"
    r")",
    re.IGNORECASE,
)

_PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"complete numbered mechanical steps|"
    r"every formula needed, copied verbatim|"
    r"comprehensive explanation tying excerpts|"
    r"Mechanical steps for solving|"
    r"List of specific mechanical steps|"
    r"Knowledge of statistical formulas such as the t-test|"
    r"Formulas for statistical methods|"
    r"Definitions of statistical terms|"
    r"explanation tying excerpts to this question|"
    r"quotes\\_used|Test memory from arka|my name is sumit"
    r")",
    re.IGNORECASE,
)

_JUNK_RE = re.compile(
    r"(?:"
    r"Section Id\s*:|Sub-Section Id|Question Shuffling|"
    r"Memory \w+|semantic\)|mechanical process|torque|fluid dynamics|"
    r"disassembly, assembly|Newton's laws|"
    r"640653\d+|Sem2 English2|INTRODUCTION TO PYTHON"
    r")",
    re.IGNORECASE,
)

_GENERIC_CONCEPT_RE = re.compile(
    r"^(?:Mechanical steps|Formulas|Definitions|Subjects|Variables|Options)$",
    re.I,
)

_FRAGMENT_FORMULA_RE = re.compile(
    r"^(?:and n\(S\)|where P\(A\)|n!$|\(n-1\)|\(a, b\)|\(1\)|\(2\)|\(3\))",
    re.I,
)

TOPIC_ORDER = [
    "Descriptive Statistics",
    "Probability & Conditional Probability",
    "Combinatorics & Counting",
    "Correlation & Association",
    "Data Types & Study Design",
    "Frequency & Distribution",
    "Transformations of Data",
    "Exam Skills (MCQ / MSQ / SA)",
]

_TOPIC_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "Probability & Conditional Probability",
        re.compile(
            r"probability|bayes|P\s*\(|conditional|independent|engine issues|"
            r"blood test|ALS|taxi|co-education|principal|at least one|"
            r"same colour|black pair",
            re.I,
        ),
    ),
    (
        "Combinatorics & Counting",
        re.compile(
            r"permutation|combination|arranged|ways to|circular|ENGLISH|"
            r"committee|replacement|violinist|performers|order matters|"
            r"simultaneously",
            re.I,
        ),
    ),
    (
        "Correlation & Association",
        re.compile(
            r"correlation|scatter|association|contingency|literacy|voting|"
            r"reaction.*gender|cross.?tab",
            re.I,
        ),
    ),
    (
        "Data Types & Study Design",
        re.compile(
            r"nominal|ordinal|interval|ratio scale|cross sectional|time series|"
            r"analytics|software|core companies|statistical analysis",
            re.I,
        ),
    ),
    (
        "Frequency & Distribution",
        re.compile(
            r"frequency|cumulative|relative frequency|histogram|stem.?and.?leaf|"
            r"IQR|quartile|outlier|box.?plot",
            re.I,
        ),
    ),
    (
        "Transformations of Data",
        re.compile(
            r"subtracted|multiplied|modified marks|linear transform|"
            r"adding.*constant|scaling",
            re.I,
        ),
    ),
    (
        "Descriptive Statistics",
        re.compile(
            r"mean|median|mode|variance|standard deviation|range|average|"
            r"runner|distance|height|salary|marks|SD\s*\(|Var\s*\(",
            re.I,
        ),
    ),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _dedupe_key(text: str) -> str:
    return _normalize(re.sub(r"[^\w\s]", "", text))[:140]


def _clean_question_text(text: str) -> str:
    text = re.sub(r"^Question\s+", "", text.strip(), flags=re.I)
    text = re.split(r"\s*:\s*Yes Show Word Count", text, maxsplit=1)[0]
    text = re.split(r"\s*Answers Type\s*:", text, maxsplit=1)[0]
    text = re.sub(
        r"^Display Question Number\s*:.*?Correct Marks\s*:\s*\d+\s*",
        "",
        text,
        flags=re.I | re.DOTALL,
    )
    text = re.sub(r"^Time\s*:\s*0\s*", "", text, flags=re.I)
    text = re.sub(r"Selectable Option\s*:\s*\d+\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _is_placeholder(text: str) -> bool:
    return not text.strip() or bool(_PLACEHOLDER_RE.search(text))


def _is_junk(text: str) -> bool:
    if _is_placeholder(text) or _BOILERPLATE_RE.search(text):
        return True
    if _JUNK_RE.search(text):
        return True
    if text.count(":") >= 4 and "Question" in text:
        return True
    return False


def _formula_quality(formula: str) -> int:
    """Higher score = keep this variant when deduplicating."""
    score = len(formula)
    if "=" in formula:
        score += 20
    if _FRAGMENT_FORMULA_RE.match(formula.strip()):
        score -= 50
    return score


def _is_boilerplate(answer: StudyAnswer) -> bool:
    if answer.skipped:
        return True
    combined = f"{answer.question_text} {answer.explanation}"
    if _BOILERPLATE_RE.search(combined):
        return True
    cleaned = _clean_question_text(answer.question_text)
    if len(cleaned) < 30 and re.match(
        r"^(?:Display Question Number|Time\s*:\s*0|Max\. Selectable)",
        cleaned,
        re.I,
    ):
        return True
    if not answer.concepts and not answer.steps and _is_placeholder(answer.explanation):
        return True
    if answer.concepts and all(
        _is_placeholder(c.name) and _is_placeholder(c.description) for c in answer.concepts
    ):
        return True
    return False


def _infer_topic(answer: StudyAnswer, question: ExamQuestion | None) -> str:
    blob = " ".join(
        [
            answer.question_text,
            answer.explanation,
            " ".join(f"{c.name} {c.description}" for c in answer.concepts),
            question.text if question else "",
        ]
    )
    for topic, pattern in _TOPIC_RULES:
        if pattern.search(blob):
            return topic
    return "Exam Skills (MCQ / MSQ / SA)"


def _parse_explanation(explanation: str) -> tuple[list[str], list[str], str]:
    formulas: list[str] = []
    facts: list[str] = []
    prose = explanation

    if "FORMULAS:" in prose:
        before, _, after = prose.partition("KEY FACTS:")
        for line in before.split("\n"):
            line = line.strip()
            if line.startswith("- ") and not _is_placeholder(line[2:]):
                formulas.append(line[2:].strip())
        prose = after if after else ""

    if "KEY FACTS:" in explanation and "KEY FACTS:" not in prose:
        pass
    elif prose.startswith("KEY FACTS:"):
        parts = prose.split("\n\n", 1)
        for line in parts[0].split("\n"):
            line = line.strip()
            if line.startswith("- ") and not _is_placeholder(line[2:]):
                facts.append(line[2:].strip())
        prose = parts[1] if len(parts) > 1 else ""

    if _is_placeholder(prose):
        prose = ""
    return formulas, facts, prose.strip()


def _extract_question_from_quotes(answer: StudyAnswer) -> str:
    best = ""
    for quote in answer.quotes:
        text = quote.quote
        for label in (
            "Question Label : Short Answer Question\n",
            "Question Label : Multiple Choice Question\n",
            "Question Label : Multiple Select Question\n",
        ):
            if label not in text:
                continue
            after = text.split(label, 1)[1]
            after = re.split(r"\nOptions\s*:", after, maxsplit=1)[0]
            after = re.split(r"\nResponse Type", after, maxsplit=1)[0]
            after = re.sub(r"\s+", " ", after).strip()
            if len(after) > len(best) and not _BOILERPLATE_RE.search(after):
                best = after
    return best


def _effective_question_text(answer: StudyAnswer, question: ExamQuestion | None) -> str:
    cleaned = _clean_question_text(answer.question_text)
    if question and question.text:
        q_clean = _clean_question_text(question.text)
        if len(q_clean) > len(cleaned) and not _BOILERPLATE_RE.search(q_clean):
            cleaned = q_clean
    if len(cleaned) < 40 or _BOILERPLATE_RE.search(cleaned):
        from_quotes = _extract_question_from_quotes(answer)
        if from_quotes:
            cleaned = from_quotes
    return cleaned


class _TopicBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.definitions: dict[str, tuple[str, str]] = {}
        self.formulas: dict[str, str] = {}
        self.facts: dict[str, str] = {}
        self.examples: list[str] = []
        self.methods: dict[str, str] = {}

    def add_definition(self, name: str, description: str) -> None:
        if _GENERIC_CONCEPT_RE.match(name.strip()) or _is_junk(name) or _is_junk(description):
            return
        if _is_placeholder(description) and _is_placeholder(name):
            return
        key = _dedupe_key(name)
        if key not in self.definitions:
            self.definitions[key] = (
                sanitize_plaintext(name.strip()),
                sanitize_plaintext(description.strip()),
            )

    def add_formula(self, formula: str) -> None:
        formula = sanitize_plaintext(formula.strip())
        if not formula or _is_junk(formula) or _FRAGMENT_FORMULA_RE.match(formula):
            return
        # Normalize key: strip whitespace variants of same formula
        norm = _normalize(formula.replace("×", "*"))
        norm_key = re.sub(r"\s+", "", norm)[:80]
        existing = self.formulas.get(norm_key)
        if existing is None or _formula_quality(formula) > _formula_quality(existing):
            self.formulas[norm_key] = formula

    def add_fact(self, fact: str) -> None:
        fact = sanitize_plaintext(fact.strip())
        if fact and not _is_junk(fact):
            self.facts[_dedupe_key(fact)] = fact

    def add_example(self, text: str) -> None:
        text = sanitize_plaintext(text.strip())
        if len(text) < 35 or _is_junk(text):
            return
        if text.startswith("Question Label"):
            return
        key = _dedupe_key(text)
        if key not in {_dedupe_key(e) for e in self.examples}:
            self.examples.append(text)

    def add_method(self, step: str) -> None:
        step = sanitize_plaintext(step.strip())
        if step and not _is_junk(step) and len(step) > 20:
            self.methods[_dedupe_key(step)] = step

    def is_empty(self) -> bool:
        return not any(
            [self.definitions, self.formulas, self.facts, self.examples, self.methods]
        )


def _render_topic(bucket: _TopicBucket) -> list[str]:
    lines: list[str] = [bucket.name, "-" * len(bucket.name), ""]

    if bucket.definitions:
        lines.append(
            "The following definitions and relationships come up repeatedly in the exam papers."
        )
        lines.append("")
        for _key, (name, desc) in sorted(bucket.definitions.items(), key=lambda x: x[1][0].lower()):
            if not desc or len(desc) < 10:
                lines.append(f"{name} appears frequently in exam questions on this topic.")
                continue
            desc_clean = desc.strip()
            name_lower = name.strip().lower()
            desc_lower = desc_clean.lower()
            if desc_lower.startswith(name_lower):
                lines.append(f"{desc_clean.rstrip('.')}.")
            elif desc_lower.startswith("the " + name_lower) or desc_lower.startswith("a " + name_lower):
                lines.append(f"{desc_clean.rstrip('.')}.")
            elif desc_lower.startswith(("calculate", "understand", "know", "find", "determine")):
                lines.append(f"{name}: {desc_clean.rstrip('.')}.")
            else:
                lines.append(f"{name} — {desc_clean.rstrip('.')}.")
        lines.append("")

    if bucket.formulas:
        lines.append("Key formulas (verbatim from source material):")
        lines.append("")
        seen_norm: set[str] = set()
        for formula in sorted(bucket.formulas.values(), key=lambda f: (-_formula_quality(f), f.lower())):
            norm = re.sub(r"\s+", "", _normalize(formula.split("=")[0] if "=" in formula else formula))
            if norm in seen_norm and len(formula) < 40:
                continue
            seen_norm.add(norm)
            lines.extend(render_formula_lines(formula, bullet="  • ", indent="    "))
        lines.append("")

    if bucket.facts:
        lines.append("Important facts and conditions to remember:")
        lines.append("")
        for fact in sorted(bucket.facts.values(), key=str.lower)[:12]:
            lines.append(f"  • {fact.rstrip('.')}.")
        lines.append("")

    if bucket.methods:
        lines.append("Conceptual reasoning from exam questions:")
        lines.append("")
        top_methods = sorted(bucket.methods.values(), key=lambda s: -len(s))[:15]
        for step in top_methods:
            lines.append(f"  • {step}")
        lines.append("")

    if bucket.examples:
        lines.append("Representative exam scenarios:")
        lines.append("")
        seen: set[str] = set()
        for ex in bucket.examples:
            key = _dedupe_key(ex)
            if key in seen:
                continue
            seen.add(key)
            if ex.endswith("."):
                lines.append(f"  • {ex}")
            else:
                lines.append(f"  • {ex}.")
        lines.append("")

    return lines


def _formula_line(formula) -> str:
    """Build display string from FormulaItem or legacy string."""
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


def _ingest_answer(
    bucket: _TopicBucket,
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

    extra_formulas, facts, prose = _parse_explanation(answer.explanation)
    for formula in extra_formulas:
        bucket.add_formula(formula)
    for fact in facts:
        bucket.add_fact(fact)
    if prose:
        bucket.add_example(prose)

    for reason in answer.reasoning:
        bucket.add_method(reason)

    for step in answer.steps:
        bucket.add_method(step)

    qtext = _effective_question_text(answer, question)
    if qtext:
        bucket.add_example(qtext)


def render_summary_txt(
    answers: list[StudyAnswer],
    questions: list[ExamQuestion] | None = None,
) -> str:
    """Render study answers as cohesive topic-organized prose summary."""
    by_id = {q.question_id: q for q in (questions or [])}
    buckets: dict[str, _TopicBucket] = {}

    kept = 0
    skipped = 0
    for answer in answers:
        if _is_boilerplate(answer):
            skipped += 1
            continue
        kept += 1
        topic = _infer_topic(answer, by_id.get(answer.question_id))
        if topic not in buckets:
            buckets[topic] = _TopicBucket(topic)
        _ingest_answer(buckets[topic], answer, by_id.get(answer.question_id))

    lines = [
        "STATISTICS FOR DATA SCIENCE — COMPREHENSIVE STUDY SUMMARY",
        "=" * 58,
        "",
        "This guide synthesizes material from five Semester I Statistics exam papers.",
        f"It consolidates {kept} substantive topics ({skipped} boilerplate/metadata entries omitted).",
        "Content is grouped by statistical theme rather than by individual question number.",
        "",
    ]

    ordered_topics = [t for t in TOPIC_ORDER if t in buckets and not buckets[t].is_empty()]
    for topic in ordered_topics:
        lines.extend(_render_topic(buckets[topic]))
        lines.append("")

    if not ordered_topics:
        lines.append("No substantive content found after filtering boilerplate entries.")

    return "\n".join(lines).rstrip() + "\n"


def load_answers_from_json(path: Path) -> list[StudyAnswer]:
    data = json.loads(path.read_text())
    return [StudyAnswer.model_validate(item) for item in data]


def load_questions_from_jsonl(path: Path) -> list[ExamQuestion]:
    questions: list[ExamQuestion] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            ExamQuestion(
                question_id=row["question_id"],
                exam_id=row.get("exam_id", ""),
                exam_path=row.get("exam_path", ""),
                number=str(row.get("number", "")),
                text=row.get("text", ""),
                topic=row.get("topic", ""),
            )
        )
    return questions


def export_summary_file(
    json_path: Path,
    output_path: Path,
    questions_path: Path | None = None,
) -> Path:
    answers = load_answers_from_json(json_path)
    questions = load_questions_from_jsonl(questions_path) if questions_path else None
    text = render_summary_txt(answers, questions)
    output_path.write_text(text)
    return output_path
