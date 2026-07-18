from __future__ import annotations

from pathlib import Path

from summarize_pdfs.config import AppConfig
from summarize_pdfs.models import TextChunk


class Embedder:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=len(texts) > 128,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]


class VectorStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedder = Embedder(config)
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            import chromadb

            self.config.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def indexed_source_ids(self) -> set[str]:
        result = self.collection.get(include=[])
        ids = result.get("ids") or []
        source_ids: set[str] = set()
        if ids:
            meta = self.collection.get(ids=ids, include=["metadatas"])
            for md in meta.get("metadatas") or []:
                if md and "source_id" in md:
                    source_ids.add(md["source_id"])
        return source_ids

    def upsert_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0

        existing = set(self.collection.get(ids=[c.chunk_id for c in chunks], include=[]).get("ids") or [])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        embeddings = self.embedder.embed_texts(texts)

        self.collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source_id": c.source_id,
                    "source_path": c.source_path,
                    "doc_type": c.doc_type.value,
                    "page": c.page,
                    "topic": c.topic or "",
                    "char_count": c.char_count,
                }
                for c in new_chunks
            ],
        )
        return len(new_chunks)

    def query(
        self,
        query_text: str,
        *,
        top_k: int | None = None,
        doc_type: str | None = "textbook",
        topic: str | None = None,
    ) -> list[dict]:
        top_k = top_k or self.config.top_k
        embedding = self.embedder.embed_texts([query_text])[0]

        where: dict | None = None
        if doc_type:
            where = {"doc_type": doc_type}
        if topic:
            clause = {"topic": topic}
            where = {"$and": [where, clause]} if where else clause

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "score": 1.0 - float(dist),
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
