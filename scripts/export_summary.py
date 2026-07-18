#!/usr/bin/env python3
"""Standalone export script — no package install required."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from summarize_pdfs.export.summary import export_summary_file  # noqa: E402

DEFAULT_JSON = Path("/Users/sumitmishra/dev/stats/output/study_guide_20260718T135119Z.json")
DEFAULT_OUT = Path("/Users/sumitmishra/dev/stats/output/study_guide_complete.txt")
DEFAULT_QUESTIONS = Path("/Users/sumitmishra/dev/stats/processed/questions.jsonl")


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    questions = DEFAULT_QUESTIONS if DEFAULT_QUESTIONS.exists() else None
    result = export_summary_file(json_path, out_path, questions)
    print(f"Wrote {result} ({result.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
