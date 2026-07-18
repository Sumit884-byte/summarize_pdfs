from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from summarize_pdfs.config import Settings, load_config
from summarize_pdfs.eval.gold_standard import run_eval_from_json, summarize_eval
from summarize_pdfs.exam.question_parser import load_questions
from summarize_pdfs.export.notes import export_notes_file
from summarize_pdfs.export.summary import export_summary_file
from summarize_pdfs.index.vector_store import VectorStore
from summarize_pdfs.memory.store import StudyMemory
from summarize_pdfs.pipeline.expand_notes import expand_study_notes
from summarize_pdfs.pipeline.polish_notes import polish_study_notes
from summarize_pdfs.pipeline.llm import llm_is_configured
from summarize_pdfs.pipeline.runner import (
    build_concept_graph_from_questions,
    generate_study_guide,
    index_textbooks,
    parse_exams,
    produce_study_materials,
    run_full_pipeline,
)
from summarize_pdfs.pipeline.cooccurrence import load_concept_graph, render_cluster_section

app = typer.Typer(
    name="summarize-pdfs",
    help="Exam-guided RAG pipeline — scale textbook study with quiz papers as ground truth.",
)
console = Console()
memory_app = typer.Typer(help="Supermemory study memory (local or cloud)")
app.add_typer(memory_app, name="memory")


@app.command("init")
def init_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Create config and data directories."""
    example = Path("config.example.yaml")
    if not config_path.exists() and example.exists():
        shutil.copy(example, config_path)
        console.print(f"Created {config_path} from config.example.yaml")
    config = load_config(config_path)
    config.ensure_dirs()
    console.print(f"Data dirs ready under [bold]{config.data_dir}[/bold]")
    console.print("Drop PDFs into:")
    console.print(f"  • {config.source_dir / 'textbooks'}")
    console.print(f"  • {config.source_dir / 'exams'}")


@app.command("index")
def index_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Chunk and embed textbook PDFs into the vector store."""
    config = load_config(config_path)
    new_chunks, warnings = index_textbooks(config)
    total = VectorStore(config).count()
    console.print(f"Indexed [bold]{new_chunks}[/bold] new chunks ([bold]{total}[/bold] total)")
    for w in warnings[:5]:
        console.print(f"[yellow]![/yellow] {w}")


@app.command("concept-graph")
def concept_graph_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force re-extraction of concepts"),
) -> None:
    """Build or display co-occurrence map from parsed exam questions."""
    config = load_config(config_path)
    if not llm_is_configured(config, Settings()):
        console.print("[red]No LLM configured — cannot extract concepts[/red]")
        raise typer.Exit(1)

    questions = load_questions(config.questions_db)
    if not questions:
        console.print("[red]No questions found — run parse-exams first[/red]")
        raise typer.Exit(1)

    if rebuild or not config.concept_graph_path.exists():
        asyncio.run(build_concept_graph_from_questions(config, questions))
    else:
        console.print(f"[dim]Using existing graph at {config.concept_graph_path}[/dim]")

    graph = load_concept_graph(config.concept_graph_path)
    if graph is None:
        console.print("[red]Failed to load concept graph[/red]")
        raise typer.Exit(1)

    table = Table(title="Concept Co-occurrence Clusters")
    table.add_column("Cluster")
    table.add_column("Questions", justify="right")
    for cluster in graph.concept_clusters[:20]:
        table.add_row(cluster.display_name, str(cluster.question_count))
    console.print(table)

    if graph.pair_counts:
        console.print(f"\nTop co-occurring pairs (threshold ≥ {graph.threshold}):")
        pairs = sorted(graph.pair_counts.items(), key=lambda item: (-item[1], item[0]))[:15]
        for key, count in pairs:
            left, right = key.split("|||", 1)
            console.print(f"  • {left.title()} + {right.title()} ({count})")

    console.print(f"\nSaved → {config.concept_graph_path}")
    for line in render_cluster_section(graph):
        if line.startswith("•"):
            console.print(line)


@app.command("parse-exams")
def parse_exams_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use regex parser instead of LLM"),
) -> None:
    """Extract structured questions from exam PDFs."""
    config = load_config(config_path)
    if not llm_is_configured(config, Settings()) and not no_llm:
        console.print("[yellow]No LLM configured — falling back to heuristic parser[/yellow]")
        no_llm = True
    questions = parse_exams(config, use_llm=not no_llm)
    console.print(f"Parsed [bold]{len(questions)}[/bold] questions → {config.questions_db}")


@app.command("generate")
def generate_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    topic: str | None = typer.Option(None, "--topic", "-t"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
) -> None:
    """Run two-step RAG: concepts → retrieval → quoted synthesis."""
    config = load_config(config_path)
    answers, out_path = asyncio.run(generate_study_guide(config, topic=topic, limit=limit))
    console.print(f"Generated [bold]{len(answers)}[/bold] answers → {out_path}")
    console.print(f"Exported study_guide_complete.txt and study_notes.txt in {config.output_dir}")


@app.command("produce")
def produce_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    topic: str | None = typer.Option(None, "--topic", "-t"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    no_llm_parse: bool = typer.Option(False, "--no-llm-parse"),
    no_expand: bool = typer.Option(False, "--no-expand", help="Skip textbook expansion step"),
    polish: bool = typer.Option(False, "--polish", help="Run final LLM polish on study notes"),
) -> None:
    """Full pipeline: index → parse → generate → export → expand with textbook."""
    config = load_config(config_path)
    produce_study_materials(
        config,
        use_llm_parse=not no_llm_parse,
        topic=topic,
        limit=limit,
        expand=not no_expand,
        polish=polish,
    )


@app.command("expand-notes")
def expand_notes_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    json_path: Path | None = typer.Option(None, "--json", help="Study guide JSON"),
    notes_output: Path | None = typer.Option(None, "--notes", "-n", help="Output notes path"),
    summary_output: Path | None = typer.Option(None, "--summary", "-s", help="Output summary path"),
) -> None:
    """Expand study notes and summary using indexed textbook RAG."""
    config = load_config(config_path)
    notes_path, summary_path, chunk_count = asyncio.run(
        expand_study_notes(
            config,
            json_path=json_path,
            notes_path=notes_output,
            summary_path=summary_output,
        )
    )
    console.print(
        f"Done — [bold]{chunk_count}[/bold] chunks, notes → {notes_path}, summary → {summary_path}"
    )


@app.command("polish-notes")
def polish_notes_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    notes_input: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Input study notes (default: study_notes.txt in output dir)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path (default: study_notes_polished.txt)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite study_notes.txt (creates .bak backup)",
    ),
) -> None:
    """Polish study notes by topic: dedupe, clarify, and add cross-links."""
    config = load_config(config_path)
    if not llm_is_configured(config, Settings()):
        console.print("[red]No LLM configured — cannot polish notes[/red]")
        raise typer.Exit(1)
    notes_path = notes_input or config.output_dir / "study_notes.txt"
    out_path = output
    if out_path is None and overwrite:
        out_path = notes_path
    result = asyncio.run(
        polish_study_notes(
            config,
            notes_path=notes_path,
            output_path=out_path,
            overwrite=overwrite,
        )
    )
    console.print(f"Done → {result}")


@app.command("dedupe-notes")
def dedupe_notes_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    notes_input: Path | None = typer.Option(None, "--input", "-i"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Remove duplicate formulas/definitions and junk lines from study notes."""
    from summarize_pdfs.export.dedupe_notes import deduplicate_study_notes

    config = load_config(config_path)
    notes_path = notes_input or config.output_dir / "study_notes.txt"
    if not notes_path.exists():
        console.print(f"[red]Not found: {notes_path}[/red]")
        raise typer.Exit(1)
    out_path = output or (notes_path if overwrite else config.output_dir / "study_notes_deduped.txt")
    text = deduplicate_study_notes(notes_path.read_text())
    if overwrite:
        backup = notes_path.with_suffix(notes_path.suffix + ".bak")
        if not backup.exists() or notes_path.stat().st_mtime > backup.stat().st_mtime:
            import shutil
            shutil.copy2(notes_path, backup)
    out_path.write_text(text)
    lines = len(text.splitlines())
    console.print(f"Deduplicated notes ({lines} lines) → {out_path}")


@app.command("export-summary")
def export_summary(
    json_path: Path | None = typer.Option(
        None,
        "--json",
        help="Study guide JSON (defaults to latest in output dir)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .txt path (default: study_guide_complete.txt in output dir)",
    ),
    config_path: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        help="Path to config.yaml",
    ),
) -> None:
    """Export a topic-organized prose summary from structured study guide JSON."""
    config = load_config(config_path)
    out_dir = config.output_dir

    if json_path is None:
        candidates = sorted(out_dir.glob("study_guide_*.json"), reverse=True)
        if not candidates:
            typer.echo(f"No study_guide_*.json found in {out_dir}", err=True)
            raise typer.Exit(1)
        json_path = candidates[0]

    if output is None:
        output = out_dir / "study_guide_complete.txt"

    questions_path = config.processed_dir / "questions.jsonl"
    if not questions_path.exists():
        questions_path = None

    result = export_summary_file(json_path, output, questions_path)
    typer.echo(f"Summary written → {result}")


@app.command("export-notes")
def export_notes(
    json_path: Path | None = typer.Option(
        None,
        "--json",
        help="Study guide JSON (defaults to latest in output dir)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output .txt path (default: study_notes.txt in output dir)",
    ),
    config_path: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        help="Path to config.yaml",
    ),
) -> None:
    """Export compact study notes (formulas, tricks, patterns) from study guide JSON."""
    config = load_config(config_path)
    out_dir = config.output_dir

    if json_path is None:
        candidates = sorted(out_dir.glob("study_guide_*.json"), reverse=True)
        if not candidates:
            typer.echo(f"No study_guide_*.json found in {out_dir}", err=True)
            raise typer.Exit(1)
        json_path = candidates[0]

    if output is None:
        output = out_dir / "study_notes.txt"

    questions_path = config.processed_dir / "questions.jsonl"
    if not questions_path.exists():
        questions_path = None

    result = export_notes_file(json_path, output, questions_path)
    typer.echo(f"Notes written → {result}")


@app.command("run")
def run_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    no_llm_parse: bool = typer.Option(False, "--no-llm-parse"),
) -> None:
    """Full pipeline: index → parse exams → generate study guide."""
    config = load_config(config_path)
    run_full_pipeline(config, use_llm_parse=not no_llm_parse)


@app.command("eval")
def eval_cmd(
    answers_json: Path = typer.Argument(..., help="study_guide_*.json from generate/run"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Score generated answers against official marking schemes."""
    config = load_config(config_path)
    results = asyncio.run(run_eval_from_json(answers_json, config))
    summary = summarize_eval(results)

    table = Table(title="Gold Standard Evaluation")
    table.add_column("Question")
    table.add_column("Score")
    table.add_column("Pass")
    for r in results:
        table.add_row(r.question_id, f"{r.score:.2f}", "✓" if r.passed else "✗")
    console.print(table)
    console.print(summary)

    out = config.output_dir / f"eval_{answers_json.stem}.json"
    out.write_text(json.dumps({"summary": summary, "results": [r.model_dump() for r in results]}, indent=2))
    console.print(f"Saved → {out}")


@app.command("status")
def status_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Show index and question DB stats."""
    config = load_config(config_path)
    store = VectorStore(config)
    questions = load_questions(config.questions_db)
    exams = list((config.source_dir / "exams").glob("**/*.pdf"))
    books = list((config.source_dir / "textbooks").glob("**/*.pdf"))

    table = Table(title="Pipeline Status")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Textbook PDFs", str(len(books)))
    table.add_row("Exam PDFs", str(len(exams)))
    table.add_row("Indexed chunks", str(store.count()))
    table.add_row("Parsed questions", str(len(questions)))

    mem = StudyMemory(config)
    sm = mem.status()
    table.add_row("Supermemory", "enabled" if sm.get("enabled") and sm.get("available") else "off")
    if sm.get("available"):
        table.add_row("Memory mode", str(sm.get("effective_mode", sm.get("mode", "?"))))
        table.add_row("Local memories", str(sm.get("local_entries", 0)))
    console.print(table)


@memory_app.command("status")
def memory_status_cmd(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Show Supermemory backend and local cache stats."""
    config = load_config(config_path)
    sm = StudyMemory(config).status()
    table = Table(title="Supermemory Status")
    table.add_column("Setting")
    table.add_column("Value")
    for key, value in sm.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


@memory_app.command("recall")
def memory_recall_cmd(
    query: str = typer.Argument(..., help="Search prior study memories"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    limit: int = typer.Option(5, "--limit", "-n"),
) -> None:
    """Search stored study memories."""
    config = load_config(config_path)
    memory = StudyMemory(config)
    if not memory.enabled:
        console.print("[yellow]Supermemory disabled or arka not installed[/yellow]")
        raise typer.Exit(1)
    hits = memory.recall(query, limit=limit)
    if not hits:
        console.print("No matching memories.")
        raise typer.Exit(0)
    for line in hits:
        console.print(f"• {line}")


if __name__ == "__main__":
    app()
