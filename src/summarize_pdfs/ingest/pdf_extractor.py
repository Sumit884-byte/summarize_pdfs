from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fitz

from summarize_pdfs.models import DocType, OCRReport, TextChunk


def source_id(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return f"{path.stem}_{digest}"


def extract_pages(path: Path) -> list[tuple[int, str]]:
    doc = fitz.open(path)
    pages: list[tuple[int, str]] = []
    try:
        for idx in range(len(doc)):
            page = doc[idx]
            text = page.get_text("text").strip()
            pages.append((idx + 1, text))
    finally:
        doc.close()
    return pages


def ocr_sanity_check(path: Path, threshold: float = 0.55) -> OCRReport:
    pages = extract_pages(path)
    char_counts = [len(text) for _, text in pages]
    avg_chars = sum(char_counts) / max(len(char_counts), 1)

    suspect: list[int] = []
    warnings: list[str] = []

    for page_num, text in pages:
        if len(text) < 40:
            suspect.append(page_num)
            continue
        # High ratio of non-alphanumeric chars often means OCR garbage
        if text:
            alnum = sum(ch.isalnum() or ch.isspace() for ch in text)
            ratio = alnum / len(text)
            if ratio < threshold:
                suspect.append(page_num)

    is_likely_scan = avg_chars < 120 or len(suspect) > len(pages) * 0.3
    if is_likely_scan:
        warnings.append(
            "Low text density or many suspect pages — consider OCR preprocessing."
        )
    if suspect:
        warnings.append(f"Review pages: {suspect[:20]}{'...' if len(suspect) > 20 else ''}")

    return OCRReport(
        source_path=str(path),
        total_pages=len(pages),
        suspect_pages=suspect,
        avg_chars_per_page=avg_chars,
        is_likely_scan=is_likely_scan,
        warnings=warnings,
    )


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer breaking at paragraph or sentence boundary
            window = text[start:end]
            break_at = max(window.rfind("\n\n"), window.rfind(". "))
            if break_at > chunk_size * 0.5:
                end = start + break_at + (2 if window[break_at : break_at + 2] == ". " else 0)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def ingest_pdf(
    path: Path,
    doc_type: DocType,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[TextChunk], OCRReport]:
    sid = source_id(path)
    pages = extract_pages(path)
    ocr = ocr_sanity_check(path)

    chunks: list[TextChunk] = []
    chunk_idx = 0
    for page_num, page_text in pages:
        for para in _split_paragraphs(page_text):
            for piece in chunk_text(
                para, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ):
                chunk_idx += 1
                chunks.append(
                    TextChunk(
                        chunk_id=f"{sid}_p{page_num}_c{chunk_idx}",
                        source_id=sid,
                        source_path=str(path.resolve()),
                        doc_type=doc_type,
                        page=page_num,
                        text=piece,
                        char_count=len(piece),
                    )
                )
    return chunks, ocr


def save_manifest(chunks: list[TextChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def load_manifest(path: Path) -> list[TextChunk]:
    if not path.exists():
        return []
    chunks: list[TextChunk] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(TextChunk.model_validate_json(line))
    return chunks


def write_ocr_report(report: OCRReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2))
