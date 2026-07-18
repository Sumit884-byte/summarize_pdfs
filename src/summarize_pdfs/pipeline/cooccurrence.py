from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from summarize_pdfs.models import ConceptCluster, ConceptExtraction, ConceptGraph

_PAIR_SEP = "|||"


def normalize_concept(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def pair_key(a: str, b: str) -> str:
    left, right = sorted((normalize_concept(a), normalize_concept(b)))
    if left == right:
        return ""
    return f"{left}{_PAIR_SEP}{right}"


def parse_pair_key(key: str) -> tuple[str, str]:
    left, right = key.split(_PAIR_SEP, 1)
    return left, right


def _groups_from_extraction(extraction: ConceptExtraction) -> list[list[str]]:
    groups: list[list[str]] = []
    for group in extraction.co_occurring_groups:
        names = [c.strip() for c in group if c and c.strip()]
        if len(names) >= 2:
            groups.append(names)
    if groups:
        return groups

    names = [c.name.strip() for c in extraction.concepts if c.name.strip()]
    if len(names) >= 2:
        return [names]
    return []


def build_concept_graph(
    extractions: list[ConceptExtraction],
    *,
    threshold: int = 2,
) -> ConceptGraph:
    pair_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    group_labels: dict[str, list[str]] = {}
    question_groups: dict[str, list[list[str]]] = {}

    for extraction in extractions:
        groups = _groups_from_extraction(extraction)
        if groups:
            question_groups[extraction.question_id] = groups

        for group in groups:
            normalized = sorted({normalize_concept(name) for name in group})
            if len(normalized) < 2:
                continue
            group_key = "|".join(normalized)
            group_counts[group_key] += 1
            group_labels[group_key] = [name for name in group]

            for a, b in combinations(normalized, 2):
                key = pair_key(a, b)
                if key:
                    pair_counts[key] += 1

    clusters: list[ConceptCluster] = []
    for group_key, count in sorted(group_counts.items(), key=lambda item: (-item[1], item[0])):
        if count < threshold and len(group_counts) > 1:
            continue
        labels = group_labels.get(group_key, group_key.split("|"))
        display = " + ".join(_title_concept(label) for label in labels)
        clusters.append(
            ConceptCluster(
                concepts=labels,
                display_name=display,
                question_count=count,
            )
        )

    if not clusters and pair_counts:
        clusters = _clusters_from_pairs(pair_counts, threshold=threshold)

    return ConceptGraph(
        pair_counts=dict(pair_counts),
        concept_clusters=clusters,
        question_groups=question_groups,
        threshold=threshold,
    )


def _title_concept(name: str) -> str:
    normalized = normalize_concept(name)
    if not normalized:
        return name.strip()
    return " ".join(word.capitalize() for word in normalized.split())


def _clusters_from_pairs(
    pair_counts: dict[str, int],
    *,
    threshold: int,
) -> list[ConceptCluster]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for key, count in pair_counts.items():
        if count < threshold:
            continue
        left, right = parse_pair_key(key)
        union(left, right)

    components: dict[str, set[str]] = defaultdict(set)
    for key, count in pair_counts.items():
        if count < threshold:
            continue
        left, right = parse_pair_key(key)
        root = find(left)
        components[root].update([left, right])

    clusters: list[ConceptCluster] = []
    for members in components.values():
        labels = sorted(members)
        if len(labels) < 2:
            continue
        question_count = max(
            pair_counts.get(pair_key(a, b), 0)
            for a, b in combinations(labels, 2)
        )
        display = " + ".join(_title_concept(label) for label in labels)
        clusters.append(
            ConceptCluster(
                concepts=labels,
                display_name=display,
                question_count=question_count,
            )
        )
    clusters.sort(key=lambda cluster: (-cluster.question_count, cluster.display_name))
    return clusters


def expand_concepts(
    concept_names: list[str],
    graph: ConceptGraph | None,
    *,
    threshold: int | None = None,
) -> list[str]:
    if not graph or not concept_names:
        return []

    min_count = threshold if threshold is not None else graph.threshold
    expanded: set[str] = set()
    normalized_inputs = {normalize_concept(name) for name in concept_names}

    for key, count in graph.pair_counts.items():
        if count < min_count:
            continue
        left, right = parse_pair_key(key)
        if left in normalized_inputs:
            expanded.add(right)
        if right in normalized_inputs:
            expanded.add(left)

    return sorted(expanded - normalized_inputs)


def related_concepts(
    concept_name: str,
    graph: ConceptGraph | None,
    *,
    threshold: int | None = None,
    limit: int = 5,
) -> list[tuple[str, int]]:
    if not graph:
        return []

    min_count = threshold if threshold is not None else graph.threshold
    normalized = normalize_concept(concept_name)
    related: list[tuple[str, int]] = []

    for key, count in graph.pair_counts.items():
        if count < min_count:
            continue
        left, right = parse_pair_key(key)
        if left == normalized:
            related.append((right, count))
        elif right == normalized:
            related.append((left, count))

    related.sort(key=lambda item: (-item[1], item[0]))
    return related[:limit]


def save_concept_graph(graph: ConceptGraph, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph.model_dump(), indent=2))
    return path


def save_concept_extractions(extractions: list[ConceptExtraction], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item.model_dump()) for item in extractions]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def load_concept_extractions(path: Path) -> list[ConceptExtraction]:
    if not path.exists():
        return []
    extractions: list[ConceptExtraction] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        extractions.append(ConceptExtraction.model_validate(json.loads(line)))
    return extractions


def load_concept_graph(path: Path) -> ConceptGraph | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ConceptGraph.model_validate(data)


def render_cluster_section(graph: ConceptGraph) -> list[str]:
    if not graph.concept_clusters:
        return []

    lines = [
        "EXAM CONCEPT CLUSTERS (from co-occurrence analysis)",
        "-" * 52,
    ]
    for cluster in graph.concept_clusters:
        noun = "question" if cluster.question_count == 1 else "questions"
        lines.append(
            f"• {cluster.display_name} (appears together in {cluster.question_count} {noun})"
        )
    lines.append("")
    return lines


def cluster_topic_order(graph: ConceptGraph | None, fallback: list[str]) -> list[str]:
    """Order export topics by co-occurrence cluster themes, then fallback order."""
    if not graph or not graph.concept_clusters:
        return list(fallback)

    ordered: list[str] = []
    seen: set[str] = set()

    for cluster in graph.concept_clusters:
        theme = _cluster_theme(cluster)
        if theme and theme not in seen:
            ordered.append(theme)
            seen.add(theme)

    for topic in fallback:
        if topic not in seen:
            ordered.append(topic)
            seen.add(topic)
    return ordered


def _cluster_theme(cluster: ConceptCluster) -> str | None:
    joined = " ".join(cluster.concepts)
    lower = joined.lower()

    if any(word in lower for word in ("probability", "bayes", "conditional")):
        return "Probability & Conditional Probability"
    if any(word in lower for word in ("permutation", "combination", "counting")):
        return "Combinatorics & Counting"
    if any(word in lower for word in ("correlation", "association", "contingency")):
        return "Correlation & Association"
    if any(word in lower for word in ("iqr", "quartile", "outlier", "frequency")):
        return "Frequency & Distribution"
    if any(word in lower for word in ("mean", "median", "variance", "deviation")):
        return "Descriptive Statistics"
    if any(word in lower for word in ("nominal", "ordinal", "interval", "ratio")):
        return "Data Types & Study Design"
    if any(word in lower for word in ("transform", "linear", "scaled")):
        return "Transformations of Data"
    return None
