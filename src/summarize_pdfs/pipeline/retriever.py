from __future__ import annotations

from summarize_pdfs.config import AppConfig
from summarize_pdfs.index.vector_store import VectorStore
from summarize_pdfs.models import ConceptExtraction, ConceptGraph, ExamQuestion, RetrievedQuote
from summarize_pdfs.pipeline.cooccurrence import expand_concepts, normalize_concept


def _co_occurring_queries(concepts: ConceptExtraction) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    for group in concepts.co_occurring_groups:
        for name in group:
            normalized = normalize_concept(name)
            if normalized in seen:
                continue
            seen.add(normalized)
            queries.append(f"{name.strip()} definition formula statistics")

    return queries


def retrieve_for_question(
    question: ExamQuestion,
    concepts: ConceptExtraction,
    store: VectorStore,
    config: AppConfig,
    *,
    concept_graph: ConceptGraph | None = None,
) -> list[RetrievedQuote]:
    queries = list(concepts.search_queries)
    if not queries:
        queries = [question.text]
        for c in concepts.concepts:
            queries.append(f"{c.name}: {c.description}")

    queries.extend(_co_occurring_queries(concepts))

    concept_names = [c.name for c in concepts.concepts]
    for related in expand_concepts(
        concept_names,
        concept_graph,
        threshold=config.cooccurrence_threshold,
    ):
        queries.append(f"{related} definition formula statistics textbook")

    seen: set[str] = set()
    quotes: list[RetrievedQuote] = []

    for query in queries:
        hits = store.query(
            query,
            top_k=config.top_k,
            doc_type="textbook",
        )
        for hit in hits:
            chunk_id = hit["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            meta = hit["metadata"] or {}
            quotes.append(
                RetrievedQuote(
                    chunk_id=chunk_id,
                    source_path=meta.get("source_path", ""),
                    page=int(meta.get("page", 0)),
                    quote=hit["text"],
                    relevance_score=float(hit["score"]),
                )
            )
            if len(quotes) >= config.max_chunks_per_topic:
                break
        if len(quotes) >= config.max_chunks_per_topic:
            break

    quotes.sort(key=lambda q: q.relevance_score, reverse=True)
    return quotes[: config.max_chunks_per_topic]
