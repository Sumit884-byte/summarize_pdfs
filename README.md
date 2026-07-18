# summarize-pdfs

Exam-guided RAG pipeline for textbook study at scale. Uses exam papers as ground truth to retrieve and synthesize study material from indexed textbooks (OpenStax).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[arka]"
summarize-pdfs init
# Drop PDFs into data/raw/textbooks and data/raw/exams
summarize-pdfs produce --limit 3   # small test batch
summarize-pdfs produce             # full run
```

Configure LLM and data paths in `config.yaml` (see `config.example.yaml`).

## Commands

| Command | Purpose |
|---------|---------|
| `init` | Create config and data directories |
| `index` | Chunk and embed textbook PDFs |
| `parse-exams` | Extract structured questions from exam PDFs |
| `generate` | RAG synthesis with quality prompts → JSON + exports |
| `produce` | **Recommended:** index → parse → generate → export → expand |
| `expand-notes` | Enrich notes/summary with indexed textbook RAG |
| `export-summary` | Render `study_guide_complete.txt` from JSON |
| `export-notes` | Render `study_notes.txt` from JSON |
| `run` | Alias for `produce` (full pipeline) |

## Output files

- `data/output/study_guide_*.json` — structured study material per question
- `data/output/study_guide_complete.txt` — **full summary** (topic-organized prose)
- `data/output/study_notes.txt` — **quick notes** (formulas, defs, tricks only)

## Quality guarantees

Quality rules are enforced in LLM prompts (`pipeline/prompts.py`), not only at export time:

1. **Topic-organized voice** — cohesive study notes by theme, not per-question Q&A dumps
2. **Two output modes** — full summary (prose + reasoning) vs quick notes (formulas + tricks + one-line defs)
3. **Complete definitions** — full sentences, never truncated with `...`
4. **Clean plain text** — `×` not `\*`, `…` not `\...`, no LaTeX backslashes
5. **Formula glossaries** — every formula includes a `where` line defining all variables
6. **No boilerplate** — hall-ticket confirmations and section metadata skipped
7. **Textbook-grounded** — OpenStax page refs when retrieving chunks; prefer excerpt definitions
8. **Exhaustive detail** — preserve all facts and formulas from source material

Export sanitization (`export/plaintext.py`, `export/formula_glossary.py`) remains as a safety net.

## Pipeline flow

```
Exam PDFs → parse-exams → questions.jsonl
Textbook PDFs → index → Chroma vector store
questions + RAG → generate (concept extract → retrieve → synthesize)
  → study_guide_*.json
  → export summary + notes
expand-notes → textbook-backed enrichment
```

## Development

```bash
pytest tests/
```
