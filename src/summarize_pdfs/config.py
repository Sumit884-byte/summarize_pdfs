from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    provider: str = "arka"
    model: str = "gpt-4o-mini"
    max_concurrent: int = 8
    temperature: float = 0.1
    base_url: str | None = None
    task: str = "pdf"
    skill: str = "summarize_pdfs"


class SupermemoryConfig(BaseModel):
    enabled: bool = True
    mode: str = "local"  # local | auto | supermemory
    container: str = "summarize_pdfs"
    store_questions: bool = True
    store_answers: bool = True
    recall_during_rag: bool = True
    recall_limit: int = 3
    context_limit_chars: int = 2000


class AppConfig(BaseModel):
    data_dir: Path = Path("data")
    raw_dir: Path | None = None  # override: e.g. /Users/you/dev/stats
    notes_dir: Path | None = None  # study guides & notes (default: data_dir/output)
    collection_name: str = "textbook_chunks"
    chunk_size: int = 800
    chunk_overlap: int = 120
    ocr_confidence_threshold: float = 0.55
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    top_k: int = 8
    max_chunks_per_topic: int = 24
    llm: LLMConfig = Field(default_factory=LLMConfig)
    supermemory: SupermemoryConfig = Field(default_factory=SupermemoryConfig)
    exam_batch_size: int = 10
    index_workers: int = 4
    cooccurrence_threshold: int = 2

    @property
    def source_dir(self) -> Path:
        return self.raw_dir if self.raw_dir is not None else self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def output_dir(self) -> Path:
        if self.notes_dir is not None:
            return self.notes_dir
        return self.data_dir / "output"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def questions_db(self) -> Path:
        return self.processed_dir / "questions.jsonl"

    @property
    def concept_graph_path(self) -> Path:
        return self.processed_dir / "concept_graph.json"

    @property
    def concept_extractions_path(self) -> Path:
        return self.processed_dir / "concept_extractions.jsonl"

    def ensure_dirs(self) -> None:
        for path in (
            self.source_dir,
            self.source_dir / "exams",
            self.source_dir / "textbooks",
            self.processed_dir,
            self.cache_dir,
            self.output_dir,
            self.chroma_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    openai_base_url: str | None = None


def load_config(path: Path | None = None) -> AppConfig:
    if path is None:
        path = Path("config.yaml")
    if not path.exists():
        config = AppConfig()
        config.ensure_dirs()
        return config

    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if "data_dir" in raw:
        raw["data_dir"] = Path(raw["data_dir"])
    if "raw_dir" in raw and raw["raw_dir"] is not None:
        raw["raw_dir"] = Path(raw["raw_dir"])
    if "notes_dir" in raw and raw["notes_dir"] is not None:
        raw["notes_dir"] = Path(raw["notes_dir"])
    config = AppConfig.model_validate(raw)
    config.ensure_dirs()
    return config
