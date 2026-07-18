from __future__ import annotations

import os
import time
from typing import Any

from summarize_pdfs.config import AppConfig
from summarize_pdfs.models import ConceptExtraction, ExamQuestion, StudyAnswer


def _import_supermemory() -> Any | None:
    try:
        from arka.integrations import supermemory as sm

        return sm
    except ImportError:
        return None


def _effective_mode() -> str:
    raw = (os.environ.get("MEMORY") or os.environ.get("SUPERMEMORY_MODE") or "auto").strip().lower()
    if raw in ("supermemory", "cloud", "api"):
        return "supermemory"
    if raw in ("local", "offline"):
        return "local"
    return "auto"


def _local_recall(sm: Any, query: str, *, limit: int = 5) -> list[str]:
    items = sm.load_json(sm.MEMORY_FILE, [])
    if not isinstance(items, list) or not items:
        return []
    q = query.lower()
    scored: list[tuple[float, str]] = []
    for row in items:
        text = (row.get("text") or "").lower()
        tag_s = " ".join(row.get("tags") or []).lower()
        score = 0.0
        for word in q.split():
            if len(word) < 2:
                continue
            if word in text:
                score += 2.0
            if word in tag_s:
                score += 1.5
        if score > 0:
            age_days = max(0.0, (time.time() - float(row.get("ts") or 0)) / 86400)
            score += 1.0 / (1.0 + age_days / 30.0)
        if score > 0:
            scored.append((score, row.get("text") or ""))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:limit] if t.strip()]


class StudyMemory:
    """Persistent study memory via Arka's Supermemory integration (local or cloud)."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._sm = _import_supermemory()
        self._apply_env()

    @property
    def available(self) -> bool:
        return self._sm is not None

    @property
    def enabled(self) -> bool:
        return self._config.supermemory.enabled and self.available

    def _apply_env(self) -> None:
        sm_cfg = self._config.supermemory
        if sm_cfg.container and not os.environ.get("SUPERMEMORY_CONTAINER"):
            os.environ["SUPERMEMORY_CONTAINER"] = sm_cfg.container
        if sm_cfg.mode and not os.environ.get("MEMORY") and not os.environ.get("SUPERMEMORY_MODE"):
            os.environ["MEMORY"] = sm_cfg.mode

    def remember_question(self, question: ExamQuestion) -> dict[str, Any] | None:
        if not self.enabled or not self._config.supermemory.store_questions:
            return None
        topic = question.topic or "general"
        text = f"Exam Q{question.number} [{topic}] ({question.exam_id}): {question.text}"
        return self._remember(
            text,
            tags=["summarize_pdfs", "exam_question", topic, question.question_id],
            provenance={
                "kind": "exam_question",
                "question_id": question.question_id,
                "exam_id": question.exam_id,
                "number": question.number,
                "topic": topic,
            },
        )

    def remember_answer(
        self,
        question: ExamQuestion,
        answer: StudyAnswer,
    ) -> dict[str, Any] | None:
        if not self.enabled or not self._config.supermemory.store_answers:
            return None
        topic = question.topic or "general"
        concept_lines = "; ".join(f"{c.name}: {c.description}" for c in answer.concepts)
        steps = " | ".join(answer.steps[:6])
        text = (
            f"Study answer Q{question.number} [{topic}]: {question.text}\n"
            f"Concepts: {concept_lines}\n"
            f"Steps: {steps}\n"
            f"Explanation: {answer.explanation[:1200]}"
        )
        return self._remember(
            text,
            tags=["summarize_pdfs", "study_answer", topic, question.question_id],
            provenance={
                "kind": "study_answer",
                "question_id": question.question_id,
                "exam_id": question.exam_id,
                "number": question.number,
                "topic": topic,
            },
            trust_tier="project",
        )

    def context_for_question(
        self,
        question: ExamQuestion,
        concepts: ConceptExtraction | None = None,
    ) -> str:
        if not self.enabled or not self._config.supermemory.recall_during_rag:
            return ""
        parts = [question.text]
        if question.topic:
            parts.append(f"topic: {question.topic}")
        if concepts and concepts.concepts:
            parts.extend(c.name for c in concepts.concepts[:5])
        goal = " ".join(parts)
        return self._context(goal)

    def recall(self, query: str, *, limit: int | None = None) -> list[str]:
        if not self.enabled or not self._sm:
            return []
        limit = limit or self._config.supermemory.recall_limit
        return _local_recall(self._sm, query, limit=limit)

    def status(self) -> dict[str, Any]:
        sm_cfg = self._config.supermemory
        base: dict[str, Any] = {
            "enabled": sm_cfg.enabled,
            "available": self.available,
            "mode": sm_cfg.mode,
            "container": sm_cfg.container,
            "store_questions": sm_cfg.store_questions,
            "store_answers": sm_cfg.store_answers,
            "recall_during_rag": sm_cfg.recall_during_rag,
        }
        if not self.available:
            base["note"] = "Install arka: pip install -e '.[arka]'"
            return base

        items = self._sm.load_json(self._sm.MEMORY_FILE, [])
        count = len(items) if isinstance(items, list) else 0
        api_key = (os.environ.get("SUPERMEMORY_API_KEY") or os.environ.get("SUPERMEMORY_KEY") or "").strip()
        base.update(
            {
                "local_cache": str(self._sm.MEMORY_FILE),
                "local_entries": count,
                "api_key_set": bool(api_key),
                "effective_mode": _effective_mode(),
            }
        )
        return base

    def _remember(
        self,
        text: str,
        *,
        tags: list[str],
        provenance: dict[str, Any],
        trust_tier: str = "global",
    ) -> dict[str, Any]:
        assert self._sm is not None
        return self._sm.remember(
            text,
            tags=tags,
            provenance=provenance,
            trust_tier=trust_tier,
        )

    def _context(self, goal: str) -> str:
        assert self._sm is not None
        return self._sm.context_for(
            goal,
            limit_chars=self._config.supermemory.context_limit_chars,
        )
