from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from summarize_pdfs.models import ExamQuestion


def _exam_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]


_METADATA_PREFIX = re.compile(
    r"(?:"
    r"Display Question Number\s*:\s*\w+\s*"
    r"|Is Question Mandatory\s*:\s*\w+\s*"
    r"|Calculator\s*:\s*[^\s]+\s*"
    r"|Response Time\s*:\s*[^\s]+\s*"
    r"|Think Time\s*:\s*[^\s]+\s*"
    r"|Minimum Instruction\s*Time\s*:\s*\d+\s*"
    r"|Correct Marks\s*:\s*\d+\s*"
    r"|Max\. Selectable Options\s*:\s*\d+\s*"
    r"|Question Label\s*:\s*[^?]+?(?=(?:In |Find |What |Out |A |An |The |If |Choose |Distance |Manoj |Question ))"
    r"|Question Label\s*:\s*(?:Multiple Choice Question|Multiple Select Question|Comprehension|Subjective|Short Answer Question)\s*"
    r"|Response Type\s*:\s*\w+(?:\s+\w+)*\s*"
    r"|Evaluation Required For SA\s*:\s*\w+\s*"
    r"|Show Word(?:/Character)? Limit\s*:\s*[^\s]+\s*"
    r"|Choose the correct options from the following:\s*"
    r")+",
    re.IGNORECASE,
)


def _clean_body(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Strip trailing metadata blocks
    text = re.split(r"\bOptions\s*:", text, maxsplit=1)[0].strip()
    text = re.split(r"\bSub questions\b", text, maxsplit=1)[0].strip()
    # Strip repeated NPTEL metadata prefixes
    for _ in range(5):
        cleaned = _METADATA_PREFIX.sub("", text).strip()
        if cleaned == text:
            break
        text = cleaned
    # Skip boilerplate confirmation questions
    if "ARE YOU SURE YOU HAVE TO WRITE EXAM" in text.upper():
        return ""
    return text.strip()


def is_boilerplate_question(text: str) -> bool:
    """True for hall-ticket confirmations and metadata-only entries."""
    cleaned = _clean_body(text)
    if not cleaned or len(cleaned) < 20:
        return True
    upper = cleaned.upper()
    if "ARE YOU SURE YOU HAVE TO WRITE EXAM" in upper:
        return True
    if "HALL TICKET" in upper and "CONFIRM THE SUBJECTS" in upper:
        return True
    if cleaned.count(":") >= 4 and "Display Question Number" in cleaned:
        return True
    return False


def _parse_nptel_questions(text: str, exam_id: str, exam_path: str) -> list[ExamQuestion]:
    """Parser for NPTEL/CBE-style papers: 'Question Number : 76 Question Id : ...'."""
    pattern = re.compile(
        r"Question Number\s*:\s*(\d+)\s+Question Id\s*:\s*\d+\s+Question Type\s*:\s*(\w+)"
        r"(?:\s+Is Question\s+Mandatory\s*:[^\n]*)?"
        r"(?:\s+Calculator\s*:[^\n]*)?"
        r"(?:\s+Response Time\s*:[^\n]*)?"
        r"(?:\s+Think Time\s*:[^\n]*)?"
        r"(?:\s+Minimum Instruction\s*Time\s*:[^\n]*)?"
        r"(?:\s+Correct Marks\s*:\s*(\d+))?"
        r"(?:\s+Max\. Selectable Options\s*:[^\n]*)?"
        r"(?:\s+Question Label\s*:[^\n]*)?"
        r"\s*(.*?)(?=Question Number\s*:|Question Id\s*:\s*\d+\s+Question Type\s*:|Sub-Section Number\s*:|$)",
        re.DOTALL | re.IGNORECASE,
    )
    questions: list[ExamQuestion] = []
    for match in pattern.finditer(text):
        number, qtype, marks, body = match.group(1), match.group(2), match.group(3), match.group(4)
        body = _clean_body(body)
        if len(body) < 20:
            continue
        qid = f"{exam_id}_q{number}"
        questions.append(
            ExamQuestion(
                question_id=qid,
                exam_id=exam_id,
                exam_path=exam_path,
                number=number,
                text=body,
                topic=qtype.lower(),
                marks=int(marks) if marks else None,
            )
        )
    return questions


def _heuristic_questions(text: str, exam_id: str, exam_path: str) -> list[ExamQuestion]:
    """Fallback parser when LLM is unavailable."""
    nptel = _parse_nptel_questions(text, exam_id, exam_path)
    if nptel:
        return nptel

    pattern = re.compile(
        r"(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*)?(\d+[a-z]?)[\.\):]\s*(.+?)(?=(?:\n\s*(?:Q(?:uestion)?\.?\s*)?\d+[a-z]?[\.\):])|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    questions: list[ExamQuestion] = []
    for match in pattern.finditer(text):
        number, body = match.group(1), match.group(2).strip()
        body = _clean_body(body)
        if len(body) < 15:
            continue
        qid = f"{exam_id}_q{number}"
        questions.append(
            ExamQuestion(
                question_id=qid,
                exam_id=exam_id,
                exam_path=exam_path,
                number=number,
                text=body,
            )
        )
    return questions


def save_questions(questions: list[ExamQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")


def load_questions(path: Path) -> list[ExamQuestion]:
    if not path.exists():
        return []
    items: list[ExamQuestion] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(ExamQuestion.model_validate_json(line))
    return items


def merge_questions(existing: list[ExamQuestion], new: list[ExamQuestion]) -> list[ExamQuestion]:
    by_id = {q.question_id: q for q in existing}
    for q in new:
        by_id[q.question_id] = q
    return sorted(by_id.values(), key=lambda q: (q.exam_id, q.number))


async def parse_exam_with_llm(
    pages_text: str,
    exam_path: Path,
    *,
    llm_client,
    model: str,
) -> list[ExamQuestion]:
    from summarize_pdfs.pipeline.llm import chat_json
    from summarize_pdfs.pipeline.prompts import SYSTEM_EXAM_PARSE, exam_parse_user_prompt

    exam_id = _exam_id(exam_path)
    data = await chat_json(
        llm_client,
        model=model,
        prompt=exam_parse_user_prompt(pages_text),
        system_prompt=SYSTEM_EXAM_PARSE,
    )
    questions: list[ExamQuestion] = []
    for item in data.get("questions", []):
        number = str(item.get("number", len(questions) + 1))
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if "ARE YOU SURE YOU HAVE TO WRITE EXAM" in text.upper():
            continue
        qid = f"{exam_id}_q{number}"
        questions.append(
            ExamQuestion(
                question_id=qid,
                exam_id=exam_id,
                exam_path=str(exam_path.resolve()),
                number=number,
                text=text,
                topic=item.get("topic"),
                official_answer=item.get("official_answer"),
                marks=item.get("marks"),
            )
        )
    return questions


def parse_exam_pages(
    pages: list[tuple[int, str]],
    exam_path: Path,
) -> list[ExamQuestion]:
    full_text = "\n\n".join(text for _, text in pages)
    return _heuristic_questions(full_text, _exam_id(exam_path), str(exam_path.resolve()))


def group_questions_by_topic(questions: list[ExamQuestion]) -> dict[str, list[ExamQuestion]]:
    groups: dict[str, list[ExamQuestion]] = {}
    for q in questions:
        topic = (q.topic or "general").lower().strip()
        groups.setdefault(topic, []).append(q)
    return groups


def questions_to_json(questions: list[ExamQuestion]) -> str:
    return json.dumps([q.model_dump() for q in questions], indent=2)
