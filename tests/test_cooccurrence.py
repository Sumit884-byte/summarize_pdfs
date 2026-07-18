from __future__ import annotations

from unittest.mock import MagicMock

from summarize_pdfs.config import AppConfig
from summarize_pdfs.models import ConceptExtraction, ConceptGraph, ConceptItem, ExamQuestion
from summarize_pdfs.pipeline.cooccurrence import (
    build_concept_graph,
    expand_concepts,
    pair_key,
    related_concepts,
    render_cluster_section,
    save_concept_graph,
    load_concept_graph,
)
from summarize_pdfs.pipeline.retriever import retrieve_for_question


def _extraction(
    question_id: str,
    concepts: list[tuple[str, str]],
    *,
    groups: list[list[str]] | None = None,
) -> ConceptExtraction:
    return ConceptExtraction(
        question_id=question_id,
        concepts=[
            ConceptItem(name=name, description=desc)
            for name, desc in concepts
        ],
        search_queries=[f"{concepts[0][0]} query"],
        co_occurring_groups=groups or [],
    )


def test_pair_key_is_order_independent():
    assert pair_key("Bayes", "Probability") == pair_key("probability", "bayes")


def test_build_concept_graph_counts_pairs_from_groups():
    extractions = [
        _extraction(
            "q1",
            [("Probability", "chance"), ("Conditional Probability", "given event")],
            groups=[["Probability", "Conditional Probability"]],
        ),
        _extraction(
            "q2",
            [("Probability", "chance"), ("Bayes", "theorem")],
            groups=[["Probability", "Bayes"]],
        ),
        _extraction(
            "q3",
            [("Probability", "chance"), ("Conditional Probability", "given event"), ("Bayes", "theorem")],
            groups=[["Probability", "Conditional Probability", "Bayes"]],
        ),
    ]

    graph = build_concept_graph(extractions, threshold=2)

    assert graph.pair_counts[pair_key("Probability", "Conditional Probability")] >= 2
    assert graph.pair_counts[pair_key("Probability", "Bayes")] >= 2
    assert len(graph.concept_clusters) >= 1
    assert graph.concept_clusters[0].question_count >= 2


def test_build_concept_graph_fallback_to_all_concepts():
    extractions = [
        _extraction("q1", [("IQR", "spread"), ("Quartiles", "division")]),
        _extraction("q2", [("IQR", "spread"), ("Quartiles", "division")]),
    ]
    graph = build_concept_graph(extractions, threshold=2)
    assert graph.pair_counts[pair_key("IQR", "Quartiles")] == 2


def test_expand_concepts_adds_related():
    graph = ConceptGraph(
        pair_counts={
            pair_key("probability", "conditional probability"): 5,
            pair_key("probability", "bayes"): 3,
            pair_key("mean", "median"): 1,
        },
        threshold=2,
    )
    expanded = expand_concepts(["Probability"], graph)
    assert "conditional probability" in expanded
    assert "bayes" in expanded
    assert "median" not in expanded


def test_related_concepts_sorted_by_count():
    graph = ConceptGraph(
        pair_counts={
            pair_key("probability", "conditional probability"): 8,
            pair_key("probability", "bayes"): 3,
        },
        threshold=2,
    )
    related = related_concepts("Probability", graph)
    assert related[0] == ("conditional probability", 8)


def test_render_cluster_section():
    from summarize_pdfs.models import ConceptCluster

    graph = ConceptGraph(
        concept_clusters=[
            ConceptCluster(
                concepts=["probability", "conditional probability", "bayes"],
                display_name="Probability + Conditional Probability + Bayes",
                question_count=12,
            )
        ]
    )
    lines = render_cluster_section(graph)
    text = "\n".join(lines)
    assert "EXAM CONCEPT CLUSTERS" in text
    assert "12 questions" in text
    assert "Bayes" in text


def test_save_and_load_concept_graph(tmp_path):
    graph = build_concept_graph(
        [
            _extraction(
                "q1",
                [("A", "one"), ("B", "two")],
                groups=[["A", "B"]],
            )
        ],
        threshold=1,
    )
    path = tmp_path / "concept_graph.json"
    save_concept_graph(graph, path)
    loaded = load_concept_graph(path)
    assert loaded is not None
    assert loaded.pair_counts == graph.pair_counts


def test_retriever_expands_queries_with_graph():
    question = ExamQuestion(
        question_id="q1",
        exam_id="e1",
        exam_path="/tmp/exam.pdf",
        number="1",
        text="Find P(A|B) using Bayes.",
    )
    concepts = ConceptExtraction(
        question_id="q1",
        concepts=[ConceptItem(name="Probability", description="likelihood")],
        search_queries=["probability basics"],
        co_occurring_groups=[["Probability", "Conditional Probability"]],
    )
    graph = ConceptGraph(
        pair_counts={pair_key("probability", "bayes"): 4},
        threshold=2,
    )

    store = MagicMock()
    store.query.return_value = [
        {
            "chunk_id": "c1",
            "text": "Bayes theorem excerpt",
            "score": 0.9,
            "metadata": {"source_path": "/book.pdf", "page": 10},
        }
    ]

    config = AppConfig()
    quotes = retrieve_for_question(question, concepts, store, config, concept_graph=graph)

    query_texts = [call.args[0] for call in store.query.call_args_list]
    assert any("conditional probability" in q.lower() for q in query_texts)
    assert any("bayes" in q.lower() for q in query_texts)
    assert len(quotes) == 1
    assert quotes[0].page == 10
