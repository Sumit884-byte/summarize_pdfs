#!/usr/bin/env python3
"""Standalone export script — no package install required."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from summarize_pdfs.config import load_config  # noqa: E402
from summarize_pdfs.export.summary import export_summary_file  # noqa: E402


def _latest_study_guide(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("study_guide_*.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No study_guide_*.json found in {output_dir}")
    return candidates[0]


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    default_json = _latest_study_guide(config.output_dir)
    default_out = config.output_dir / "study_guide_complete.txt"
    default_questions = config.questions_db if config.questions_db.exists() else None

    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out
    questions = default_questions
    result = export_summary_file(json_path, out_path, questions)
    print(f"Wrote {result} ({result.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
