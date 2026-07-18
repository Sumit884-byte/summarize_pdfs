from __future__ import annotations

from summarize_pdfs.config import AppConfig
from summarize_pdfs.index.vector_store import VectorStore
from summarize_pdfs.models import ConceptExtraction, ExamQuestion, RetrievedQuote


def retrieve_for_question(
    question: ExamQuestion,
    concepts: ConceptExtraction,
    store: VectorStore,
    config: AppConfig,
) -> list[RetrievedQuote]:
    queries = list(concepts.search_queries)
    if not queries:
        queries = [question.text]
        for c in concepts.concepts:
            queries.append(f"{c.name}: {c.description}")

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
