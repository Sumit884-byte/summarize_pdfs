from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocType(str, Enum):
    EXAM = "exam"
    TEXTBOOK = "textbook"


class TextChunk(BaseModel):
    chunk_id: str
    source_id: str
    source_path: str
    doc_type: DocType
    page: int
    text: str
    topic: str | None = None
    char_count: int = 0


class OCRReport(BaseModel):
    source_path: str
    total_pages: int
    suspect_pages: list[int] = Field(default_factory=list)
    avg_chars_per_page: float
    is_likely_scan: bool
    warnings: list[str] = Field(default_factory=list)


class ExamQuestion(BaseModel):
    question_id: str
    exam_id: str
    exam_path: str
    number: str
    text: str
    topic: str | None = None
    official_answer: str | None = None
    marks: int | None = None


class ConceptItem(BaseModel):
    name: str
    description: str
    formulas: list[str] = Field(default_factory=list)
    has_formula: bool = False


class FormulaItem(BaseModel):
    name: str | None = None
    expression: str
    where_clause: str | None = None


class DefinitionItem(BaseModel):
    name: str
    text: str


class ConceptExtraction(BaseModel):
    question_id: str
    concepts: list[ConceptItem]
    search_queries: list[str] = Field(default_factory=list)
    co_occurring_groups: list[list[str]] = Field(default_factory=list)


class ConceptCluster(BaseModel):
    concepts: list[str]
    display_name: str
    question_count: int


class ConceptGraph(BaseModel):
    pair_counts: dict[str, int] = Field(default_factory=dict)
    concept_clusters: list[ConceptCluster] = Field(default_factory=list)
    question_groups: dict[str, list[list[str]]] = Field(default_factory=dict)
    threshold: int = 2


class RetrievedQuote(BaseModel):
    chunk_id: str
    source_path: str
    page: int
    quote: str
    relevance_score: float


class StudyAnswer(BaseModel):
    question_id: str
    question_text: str
    concepts: list[ConceptItem]
    quotes: list[RetrievedQuote]
    explanation: str
    steps: list[str] = Field(default_factory=list)
    definitions: list[DefinitionItem] = Field(default_factory=list)
    formulas: list[FormulaItem] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    tricks: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    skipped: bool = False


class EvalResult(BaseModel):
    question_id: str
    score: float
    passed: bool
    feedback: str
    generated_answer: str
    official_answer: str | None = None


class PipelineRun(BaseModel):
    run_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exam_count: int = 0
    question_count: int = 0
    chunk_count: int = 0
    output_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
