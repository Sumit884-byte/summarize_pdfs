from __future__ import annotations

import asyncio
from pathlib import Path

from summarize_pdfs.config import AppConfig, Settings
from summarize_pdfs.exam.question_parser import load_questions
from summarize_pdfs.models import EvalResult, StudyAnswer
from summarize_pdfs.pipeline.llm import chat_json, make_llm_client, map_concurrent


async def evaluate_answers(
    answers: list[StudyAnswer],
    config: AppConfig,
) -> list[EvalResult]:
    settings = Settings()
    client = make_llm_client(config, settings)
    questions = {q.question_id: q for q in load_questions(config.questions_db)}

    async def _eval_one(answer: StudyAnswer) -> EvalResult:
        official = questions.get(answer.question_id)
        if not official or not official.official_answer:
            return EvalResult(
                question_id=answer.question_id,
                score=0.0,
                passed=False,
                feedback="No official answer available for scoring.",
                generated_answer=answer.explanation,
            )

        prompt = f"""Grade the generated study answer against the official marking scheme.

Return JSON:
{{
  "score": 0.0 to 1.0,
  "passed": true if score >= 0.7,
  "feedback": "what is missing or wrong"
}}

Question:
{answer.question_text}

Official answer:
{official.official_answer}

Generated answer:
{answer.explanation}

Steps:
{chr(10).join(answer.steps)}
"""
        data = await chat_json(
            client,
            model=config.llm.model,
            prompt=prompt,
            temperature=0.0,
            cache_dir=config.cache_dir,
        )
        return EvalResult(
            question_id=answer.question_id,
            score=float(data.get("score", 0)),
            passed=bool(data.get("passed", False)),
            feedback=data.get("feedback", ""),
            generated_answer=answer.explanation,
            official_answer=official.official_answer,
        )

    return await map_concurrent(
        [_eval_one(a) for a in answers],
        limit=config.llm.max_concurrent,
    )


def summarize_eval(results: list[EvalResult]) -> dict:
    scored = [r for r in results if r.official_answer]
    if not scored:
        return {"count": 0, "avg_score": 0.0, "pass_rate": 0.0}
    avg = sum(r.score for r in scored) / len(scored)
    passed = sum(1 for r in scored if r.passed)
    return {
        "count": len(scored),
        "avg_score": round(avg, 3),
        "pass_rate": round(passed / len(scored), 3),
        "failed_ids": [r.question_id for r in scored if not r.passed],
    }


async def run_eval_from_json(answers_path: Path, config: AppConfig) -> list[EvalResult]:
    import json

    raw = json.loads(answers_path.read_text())
    answers = [StudyAnswer.model_validate(item) for item in raw]
    return await evaluate_answers(answers, config)
