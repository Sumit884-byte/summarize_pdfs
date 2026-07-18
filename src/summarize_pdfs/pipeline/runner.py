from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from summarize_pdfs.config import AppConfig, Settings, load_config
from summarize_pdfs.exam.question_parser import (
    group_questions_by_topic,
    is_boilerplate_question,
    load_questions,
    merge_questions,
    parse_exam_pages,
    parse_exam_with_llm,
    save_questions,
)
from summarize_pdfs.export.notes import export_notes_file
from summarize_pdfs.export.summary import export_summary_file
from summarize_pdfs.ingest.pdf_extractor import extract_pages, ingest_pdf, ocr_sanity_check, save_manifest, write_ocr_report
from summarize_pdfs.index.vector_store import VectorStore
from summarize_pdfs.memory.store import StudyMemory
from summarize_pdfs.models import DocType, ExamQuestion, PipelineRun, StudyAnswer, TextChunk
from summarize_pdfs.pipeline.concept_extractor import extract_concepts
from summarize_pdfs.pipeline.expand_notes import expand_study_notes
from summarize_pdfs.pipeline.polish_notes import polish_study_notes
from summarize_pdfs.pipeline.llm import llm_is_configured, make_llm_client, map_concurrent
from summarize_pdfs.pipeline.retriever import retrieve_for_question
from summarize_pdfs.pipeline.synthesizer import synthesize_answer

console = Console()


def _discover_pdfs(directory: Path) -> list[Path]:
    return sorted(directory.glob("**/*.pdf"))


def _index_pdfs(
    config: AppConfig,
    pdfs: list[Path],
    *,
    label: str,
) -> tuple[int, list[str]]:
    store = VectorStore(config)
    already = store.indexed_source_ids()
    warnings: list[str] = []
    total_new = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Indexing {label}", total=len(pdfs))
        for pdf in pdfs:
            chunks, ocr = ingest_pdf(
                pdf,
                DocType.TEXTBOOK,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            manifest = config.processed_dir / f"{pdf.stem}.jsonl"
            save_manifest(chunks, manifest)
            write_ocr_report(ocr, config.processed_dir / f"{pdf.stem}_ocr.json")
            if ocr.warnings:
                warnings.extend(ocr.warnings)

            if chunks and chunks[0].source_id not in already:
                total_new += store.upsert_chunks(chunks)
            progress.advance(task)

    return total_new, warnings


def index_textbooks(config: AppConfig) -> tuple[int, list[str]]:
    pdfs = _discover_pdfs(config.source_dir / "textbooks")
    if pdfs:
        return _index_pdfs(config, pdfs, label="textbooks")

    exam_pdfs = _discover_pdfs(config.source_dir / "exams")
    if exam_pdfs:
        console.print(
            "[yellow]No textbook PDFs — indexing exam papers as RAG source material[/yellow]"
        )
        return _index_pdfs(config, exam_pdfs, label="exam source")

    return 0, ["No PDFs found in textbooks/ or exams/"]


async def parse_exams_async(config: AppConfig, *, use_llm: bool = True) -> list[ExamQuestion]:
    pdfs = _discover_pdfs(config.source_dir / "exams")
    existing = load_questions(config.questions_db)
    all_new: list[ExamQuestion] = []

    settings = Settings()
    client = make_llm_client(config, settings) if use_llm and llm_is_configured(config, settings) else None

    if client:
        for pdf in pdfs:
            pages = extract_pages(pdf)
            ocr = ocr_sanity_check(pdf, config.ocr_confidence_threshold)
            write_ocr_report(ocr, config.processed_dir / f"{pdf.stem}_ocr.json")
            if ocr.is_likely_scan:
                console.print(
                    f"[yellow]OCR warning[/yellow] {pdf.name}: "
                    f"{ocr.warnings[0] if ocr.warnings else 'suspect scan'}"
                )
            text = "\n\n".join(t for _, t in pages)
            all_new.extend(
                await parse_exam_with_llm(text, pdf, llm_client=client, model=config.llm.model)
            )
    else:
        for pdf in pdfs:
            pages = extract_pages(pdf)
            all_new.extend(parse_exam_pages(pages, pdf))

    merged = merge_questions(existing, all_new)
    save_questions(merged, config.questions_db)

    memory = StudyMemory(config)
    if memory.enabled and config.supermemory.store_questions:
        stored = sum(1 for q in all_new if memory.remember_question(q))
        if stored:
            console.print(f"[dim]Stored {stored} exam questions in Supermemory[/dim]")

    return merged


def parse_exams(config: AppConfig, *, use_llm: bool = True) -> list[ExamQuestion]:
    return asyncio.run(parse_exams_async(config, use_llm=use_llm))


async def _process_question(
    question: ExamQuestion,
    store: VectorStore,
    client,
    config: AppConfig,
    memory: StudyMemory | None = None,
) -> StudyAnswer:
    concepts = await extract_concepts(question, client=client, config=config)
    prior_context = memory.context_for_question(question, concepts) if memory else ""
    quotes = retrieve_for_question(question, concepts, store, config)
    answer = await synthesize_answer(
        question,
        concepts,
        quotes,
        client=client,
        config=config,
        prior_context=prior_context,
    )
    if memory:
        memory.remember_answer(question, answer)
    return answer


async def generate_study_guide(
    config: AppConfig,
    *,
    topic: str | None = None,
    limit: int | None = None,
) -> tuple[list[StudyAnswer], Path]:
    settings = Settings()
    client = make_llm_client(config, settings)
    store = VectorStore(config)
    memory = StudyMemory(config)
    questions = load_questions(config.questions_db)

    if topic:
        questions = [q for q in questions if (q.topic or "general").lower() == topic.lower()]
    questions = [q for q in questions if not is_boilerplate_question(q.text)]
    if limit:
        questions = questions[:limit]

    if not questions:
        raise RuntimeError("No exam questions found. Run `parse-exams` first.")

    groups = group_questions_by_topic(questions)
    console.print(f"Processing {len(questions)} questions across {len(groups)} topics")

    coros = [_process_question(q, store, client, config, memory) for q in questions]
    answers: list[StudyAnswer] = await map_concurrent(
        coros,
        limit=config.llm.max_concurrent,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_text = render_markdown(answers, questions)
    out_path = config.output_dir / f"study_guide_{run_id}.md"
    out_path.write_text(md_text)

    json_path = config.output_dir / f"study_guide_{run_id}.json"
    json_path.write_text(
        json.dumps([a.model_dump() for a in answers], indent=2, default=str)
    )

    questions_path = config.processed_dir / "questions.jsonl"
    q_path = questions_path if questions_path.exists() else None
    complete_path = config.output_dir / "study_guide_complete.txt"
    notes_path = config.output_dir / "study_notes.txt"
    export_summary_file(json_path, complete_path, q_path)
    export_notes_file(json_path, notes_path, q_path)

    console.print(f"[green]JSON[/green] → {json_path}")
    console.print(f"[green]Summary[/green] → {complete_path}")
    console.print(f"[green]Notes[/green] → {notes_path}")
    return answers, json_path


def render_markdown(answers: list[StudyAnswer], questions: list[ExamQuestion]) -> str:
    by_id = {q.question_id: q for q in questions}
    lines = ["# Exam-Guided Study Guide", ""]
    for ans in answers:
        q = by_id.get(ans.question_id)
        num = q.number if q else "?"
        topic = q.topic if q and q.topic else "general"
        lines.extend(
            [
                f"## Q{num} — {topic}",
                "",
                f"**Question:** {ans.question_text}",
                "",
                "### Concepts to master",
            ]
        )
        for c in ans.concepts:
            formula = f" ({', '.join(c.formulas)})" if c.formulas else ""
            lines.append(f"- **{c.name}**: {c.description}{formula}")
        lines.extend(["", "### Solution steps"])
        for i, step in enumerate(ans.steps, 1):
            lines.append(f"{i}. {step}")
        lines.extend(["", "### Textbook evidence", ""])
        for quote in ans.quotes:
            lines.append(f"> p.{quote.page}: \"{quote.quote}\"")
            lines.append("")
        lines.extend(["### Explanation", "", ans.explanation, "", "---", ""])
    return "\n".join(lines)


def render_plaintext(answers: list[StudyAnswer], questions: list[ExamQuestion]) -> str:
    by_id = {q.question_id: q for q in questions}
    lines = [
        "EXAM-GUIDED COMPREHENSIVE STUDY GUIDE",
        "=" * 50,
        f"Questions covered: {len(answers)}",
        "",
    ]
    for ans in answers:
        q = by_id.get(ans.question_id)
        num = q.number if q else "?"
        topic = q.topic if q and q.topic else "general"
        lines.extend(
            [
                f"QUESTION {num} — {topic.upper()}",
                "-" * 50,
                "",
                "QUESTION:",
                ans.question_text,
                "",
                "CONCEPTS TO MASTER",
            ]
        )
        for c in ans.concepts:
            lines.append(f"  * {c.name}: {c.description}")
            for formula in c.formulas:
                lines.append(f"    Formula: {formula}")
        lines.extend(["", "SOLUTION STEPS"])
        for i, step in enumerate(ans.steps, 1):
            lines.append(f"  {i}. {step}")
        lines.extend(["", "SOURCE EVIDENCE", ""])
        for quote in ans.quotes:
            source = Path(quote.source_path).name if quote.source_path else "unknown"
            lines.append(f"  [Page {quote.page}, {source}]")
            lines.append(quote.quote)
            lines.append("")
        lines.extend(["EXPLANATION", "", ans.explanation, "", "=" * 50, ""])
    return "\n".join(lines)


def run_full_pipeline(config: AppConfig, *, use_llm_parse: bool = True) -> PipelineRun:
    return produce_study_materials(config, use_llm_parse=use_llm_parse)


async def produce_study_materials_async(
    config: AppConfig,
    *,
    use_llm_parse: bool = True,
    topic: str | None = None,
    limit: int | None = None,
    expand: bool = True,
    polish: bool = False,
) -> PipelineRun:
    """End-to-end: index → parse → generate → export → expand with textbook."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    console.print("[bold]Step 1/5[/bold] Indexing textbooks...")
    new_chunks, warnings = index_textbooks(config)
    for w in warnings[:10]:
        console.print(f"  [yellow]![/yellow] {w}")

    console.print("[bold]Step 2/5[/bold] Parsing exams...")
    questions = await parse_exams_async(config, use_llm=use_llm_parse)

    console.print("[bold]Step 3/5[/bold] Generating study guide (quality prompts)...")
    answers, json_path = await generate_study_guide(config, topic=topic, limit=limit)

    console.print("[bold]Step 4/5[/bold] Exported topic-organized summary + quick notes")

    if expand:
        console.print("[bold]Step 5/5[/bold] Expanding with indexed textbook...")
        await expand_study_notes(config, json_path=json_path)
    else:
        console.print("[bold]Step 5/5[/bold] Skipped textbook expansion")

    if polish:
        console.print("[bold]Final[/bold] Polishing study notes by topic...")
        await polish_study_notes(config)

    run = PipelineRun(
        run_id=run_id,
        exam_count=len(_discover_pdfs(config.source_dir / "exams")),
        question_count=len(questions),
        chunk_count=VectorStore(config).count(),
        output_path=str(json_path),
        metadata={
            "new_chunks_indexed": new_chunks,
            "answers_generated": len(answers),
            "expanded": expand,
            "polished": polish,
        },
    )
    manifest_path = config.output_dir / f"run_{run_id}.json"
    manifest_path.write_text(json.dumps(run.model_dump(), indent=2, default=str))
    console.print(f"[green]Done[/green] → {json_path}")
    return run


def produce_study_materials(
    config: AppConfig,
    *,
    use_llm_parse: bool = True,
    topic: str | None = None,
    limit: int | None = None,
    expand: bool = True,
    polish: bool = False,
) -> PipelineRun:
    return asyncio.run(
        produce_study_materials_async(
            config,
            use_llm_parse=use_llm_parse,
            topic=topic,
            limit=limit,
            expand=expand,
            polish=polish,
        )
    )
