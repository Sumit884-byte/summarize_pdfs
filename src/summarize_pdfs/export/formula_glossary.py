from __future__ import annotations

import re

# Short phrase definitions for common statistics notation.
_SYMBOL_GLOSSARY: dict[str, str] = {
    "IQR": "interquartile range (spread of middle 50%)",
    "Q1": "first quartile (25th percentile)",
    "Q2": "second quartile (median, 50th percentile)",
    "Q3": "third quartile (75th percentile)",
    "p": "percentile (0–100)",
    "i": "index/position in ordered data",
    "j": "lower index (floor of i when i is not an integer)",
    "percentile": "value below which a given percentage of ordered data falls",
    "P(A)": "probability of event A",
    "P(B)": "probability of event B",
    "P(A|B)": "probability of A given B has occurred",
    "P(B|A)": "probability of B given A has occurred",
    "P(A and B)": "probability of both A and B occurring",
    "P(A or B)": "probability of A or B (or both) occurring",
    "P(not A)": "probability that A does not occur",
    "P(not A or not B)": "probability that A or B (or both) do not occur",
    "P(E1)": "probability of event E1",
    "P(E2)": "probability of event E2",
    "P(E1 and E2)": "probability of both E1 and E2 occurring",
    "P(boy)": "probability of selecting a boy",
    "P(girl)": "probability of selecting a girl",
    "P(none of the events occur)": "probability that no event occurs",
    "P(no red balls are drawn)": "probability of drawing zero red balls",
    "n(A)": "number of outcomes in event A",
    "n(S)": "total number of outcomes in sample space S",
    "n": "number of items, trials, or objects",
    "k": "number chosen, successes, or selections",
    "n!": "n factorial (n × (n-1) × … × 1)",
    "C(n, k)": "combinations — ways to choose k from n without order",
    "C(n,k)": "combinations — ways to choose k from n without order",
    "σ": "standard deviation",
    "μ": "population mean",
    "x̄": "sample mean",
    "s": "sample standard deviation",
    "s²": "sample variance",
    "σ²": "population variance",
    "Var(X)": "variance of random variable X",
    "SD": "standard deviation",
    "Median": "middle value when data is ordered",
    "mean": "average of all values",
    "frequency": "count of how often a value appears",
    "cumulative frequency": "running total of frequencies up to a value",
    "total frequency": "sum of all frequencies",
    "count(value)": "number of times a value occurs",
    "frequency of previous events": "combined frequency of all prior values",
    "number of values": "sample size (count of data points)",
    "sum of squared differences from the mean": "numerator for variance (SS)",
    "P(at least one event occurs)": "probability that one or more events happen",
    "P(at least one red ball is drawn)": "probability of drawing at least one red ball",
    "P(at least one is a boy)": "probability of at least one boy",
    "P(n events)": "probability of n ordered events",
    "r": "Pearson correlation coefficient",
    "xi": "individual data value",
    "Σ": "sum over all values",
    "E(X)": "expected value of X",
    "E": "event or expected value (context-dependent)",
}

_P_VAR_RE = re.compile(r"P\([^)]+\)", re.I)
_N_FUNC_RE = re.compile(r"n\([^)]+\)", re.I)
_C_FUNC_RE = re.compile(r"C\s*\(\s*[^)]+\)", re.I)
_VAR_FUNC_RE = re.compile(r"(?:Var|SD|E)\s*\([^)]+\)", re.I)
_MULTI_SYM_RE = re.compile(
    r"\b(?:IQR|Q[123]|Median|cumulative frequency|total frequency|count\(value\))\b",
    re.I,
)
_GREEK_RE = re.compile(r"[σμ]|x̄|σ²|s²|Σ")
_SINGLE_VAR_RE = re.compile(r"\b[nkmrpij]\b")
_CHOOSE_RE = re.compile(r"\bn choose k\b", re.I)
_WORD_VAR_RE = re.compile(
    r"\b(?:frequency of previous events|number of values|sum of squared differences from the mean)\b",
    re.I,
)
_FORMULA_PREFIX_RE = re.compile(r"^[^:]+:\s*(?=[A-Za-z(])")

# Canonical formulas injected into study notes for topics where exam content may omit them.
CANONICAL_TOPIC_FORMULAS: dict[str, list[str]] = {
    "Descriptive Statistics": [
        "Percentile position: i = (p/100) × n",
        "Percentile position (alternate): i = p(n+1)/100",
        "Quartiles: Q1 = 25th percentile; Q2 = median = 50th percentile; Q3 = 75th percentile",
        (
            "Percentile interpolation: x = x_j + (i - j) × (x_next - x_j), "
            "where x = percentile value when i is not an integer; "
            "j = floor(i); x_j = data value at index j; x_next = data value at index j+1"
        ),
    ],
    "Frequency & Distribution": [
        "Percentile position: i = (p/100) × n",
        "Percentile position (alternate): i = p(n+1)/100",
        "Quartiles: Q1 = 25th percentile; Q2 = median = 50th percentile; Q3 = 75th percentile",
        (
            "Percentile interpolation: x = x_j + (i - j) × (x_next - x_j), "
            "where x = percentile value when i is not an integer; "
            "j = floor(i); x_j = data value at index j; x_next = data value at index j+1"
        ),
    ],
}


# Standard formulas keyed by concept name (fallback when LLM omits formulas).
CONCEPT_TO_FORMULA: dict[str, str] = {
    "percentile": "Percentile position: i = (p/100) × n",
    "quartile": "Quartiles: Q1 = 25th percentile; Q2 = median = 50th percentile; Q3 = 75th percentile",
    "q1": "Q1 = 25th percentile",
    "q2": "Q2 = median = 50th percentile",
    "q3": "Q3 = 75th percentile",
    "first quartile": "Q1 = 25th percentile",
    "second quartile": "Q2 = median = 50th percentile",
    "third quartile": "Q3 = 75th percentile",
    "median": "Median: middle value when n is odd; average of two middle values when n is even",
    "mean": "Mean: μ = Σxi / n",
    "sample mean": "Sample mean: x̄ = Σxi / n",
    "mode": "Mode = most frequently occurring value",
    "range": "Range = maximum - minimum",
    "variance": "Variance: σ² = Σ(xi - μ)² / N",
    "population variance": "Population variance: σ² = Σ(xi - μ)² / N",
    "sample variance": "Sample variance: s² = Σ(xi - x̄)² / (n - 1)",
    "standard deviation": "Standard deviation: σ = √(σ²)",
    "population standard deviation": "Population standard deviation: σ = √(σ²)",
    "sample standard deviation": "Sample standard deviation: s = √(s²)",
    "sd": "Standard deviation: σ = √(σ²)",
    "correlation": "Pearson correlation: r = Σ[(xi - x̄)(yi - ȳ)] / [(n-1) × sx × sy]",
    "pearson correlation": "Pearson correlation: r = Σ[(xi - x̄)(yi - ȳ)] / [(n-1) × sx × sy]",
    "pearson correlation coefficient": "Pearson correlation: r = Σ[(xi - x̄)(yi - ȳ)] / [(n-1) × sx × sy]",
    "z-score": "Z-score: z = (x - μ) / σ",
    "z score": "Z-score: z = (x - μ) / σ",
    "zscore": "Z-score: z = (x - μ) / σ",
    "iqr": "IQR = Q3 - Q1",
    "interquartile range": "IQR = Q3 - Q1",
    "probability": "Probability: P(A) = n(A) / n(S)",
    "conditional probability": "Conditional probability: P(A|B) = P(A and B) / P(B)",
    "bayes": "Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B)",
    "bayes theorem": "Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B)",
    "bayes' theorem": "Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B)",
    "combination": "Combinations: C(n, k) = n! / (k! × (n - k)!)",
    "combinations": "Combinations: C(n, k) = n! / (k! × (n - k)!)",
    "permutation": "Permutations: P(n, k) = n! / (n - k)!",
    "permutations": "Permutations: P(n, k) = n! / (n - k)!",
    "expected value": "Expected value: E(X) = Σ[xi × P(xi)]",
    "complement rule": "Complement: P(not A) = 1 - P(A)",
    "complement": "Complement: P(not A) = 1 - P(A)",
    "cumulative frequency": "Cumulative frequency = sum of frequencies up to a value",
}

_CONCEPT_ALIASES: dict[str, str] = {
    "std dev": "standard deviation",
    "std deviation": "standard deviation",
    "pearson r": "pearson correlation",
    "correlation coefficient": "correlation",
    "percentile position": "percentile",
    "percentile rank": "percentile",
}


def _normalize_concept_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9' ]", " ", name.lower())
    return re.sub(r"\s+", " ", key).strip()


def concept_has_formula(name: str) -> bool:
    """Return True if name matches a concept with a standard formula."""
    return formula_for_concept(name) is not None


def formula_for_concept(name: str) -> str | None:
    """Return the canonical formula for a concept name, or None."""
    key = _normalize_concept_key(name)
    if not key:
        return None

    alias = _CONCEPT_ALIASES.get(key)
    if alias:
        key = alias

    if key in CONCEPT_TO_FORMULA:
        return CONCEPT_TO_FORMULA[key]

    # Longest matching key wins (e.g. "sample standard deviation" before "standard deviation").
    matches = [
        (concept_key, formula)
        for concept_key, formula in CONCEPT_TO_FORMULA.items()
        if concept_key in key or key in concept_key
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    return matches[0][1]


def formulas_for_concept(name: str) -> list[str]:
    """Return all canonical formulas for a concept (usually one)."""
    formula = formula_for_concept(name)
    return [formula] if formula else []


def canonical_formulas_for_topic(topic: str) -> list[str]:
    """Return canonical formulas for a topic (empty if none)."""
    return list(CANONICAL_TOPIC_FORMULAS.get(topic, ()))


def _canonical_key(symbol: str) -> str:
    key = re.sub(r"\s+", " ", symbol.strip())
    if key.lower() == "c(n,k)":
        return "C(n, k)"
    return key


def _lookup_definition(symbol: str) -> str | None:
    key = _canonical_key(symbol)
    if key in _SYMBOL_GLOSSARY:
        return _SYMBOL_GLOSSARY[key]
    lower = key.lower()
    for glossary_key, definition in _SYMBOL_GLOSSARY.items():
        if glossary_key.lower() == lower:
            return definition
    return None


def _split_formula_and_where(formula: str) -> tuple[str, str]:
    match = re.search(r"(?<=[,;])\s*where\s+(.+)$|\s+where\s+(.+)$", formula, re.I)
    if not match:
        return formula.strip(), ""
    where = (match.group(1) or match.group(2) or "").strip()
    main = formula[: match.start()].strip().rstrip(",").strip()
    return main, where


def _parse_existing_definitions(where_clause: str) -> dict[str, str]:
    if not where_clause:
        return {}
    defs: dict[str, str] = {}
    normalized = re.sub(r",?\s+and\s+", ", ", where_clause)
    for segment in re.split(r",\s*", normalized):
        segment = segment.strip()
        if not segment:
            continue
        match = re.match(r"^(.+?)\s+(?:is|are|=)\s+(.+)$", segment, re.I)
        if match:
            var = _canonical_key(match.group(1))
            defs[var.lower()] = match.group(2).strip().rstrip(".")
    return defs


def _strip_label_prefix(formula: str) -> tuple[str, str]:
    """Return (label_prefix, formula_body) e.g. ('IQR formula', 'IQR = Q3 - Q1')."""
    if "=" not in formula:
        return "", formula
    prefix_match = _FORMULA_PREFIX_RE.match(formula)
    if not prefix_match:
        return "", formula
    label = prefix_match.group(0).rstrip(": ").strip()
    body = formula[prefix_match.end() :].strip()
    if not body or "=" not in body:
        return "", formula
    return label, body


def extract_formula_variables(formula: str) -> list[str]:
    """Extract symbolic variables from a formula, preserving order."""
    main, _where = _split_formula_and_where(formula)
    _label, body = _strip_label_prefix(main)
    expr = body or main

    found: list[str] = []
    seen: set[str] = set()

    def add(symbol: str) -> None:
        key = _canonical_key(symbol)
        norm = key.lower()
        if norm not in seen:
            seen.add(norm)
            found.append(key)

    if "=" in expr:
        lhs, rhs = expr.split("=", 1)
        lhs = lhs.strip()
        if lhs and not re.match(r"^\d", lhs):
            add(lhs)
        expr = rhs

    for pattern in (_P_VAR_RE, _N_FUNC_RE, _C_FUNC_RE, _VAR_FUNC_RE, _MULTI_SYM_RE, _GREEK_RE):
        for match in pattern.finditer(expr):
            add(match.group(0))

    if _CHOOSE_RE.search(expr):
        add("n")
        add("k")

    for match in _WORD_VAR_RE.finditer(expr):
        add(match.group(0))

    for match in _SINGLE_VAR_RE.finditer(expr):
        add(match.group(0))

    return found


def _build_where_clause(
    variables: list[str],
    existing: dict[str, str],
) -> str | None:
    parts: list[str] = []
    for var in variables:
        key = var.lower()
        if key in existing:
            parts.append(f"{var} = {existing[key]}")
            continue
        definition = _lookup_definition(var)
        if definition:
            parts.append(f"{var} = {definition}")
    if not parts:
        return None
    return "; ".join(parts)


def annotate_formula_parts(formula: str) -> tuple[str, str | None]:
    """
    Return (display_formula, where_clause_or_none).
    Preserves existing where-clauses and fills in missing variable definitions.
    """
    formula = formula.strip()
    if not formula:
        return formula, None

    label, body = _strip_label_prefix(formula)
    main, where = _split_formula_and_where(body if body else formula)
    display = f"{label}: {main}" if label else main

    existing = _parse_existing_definitions(where)
    variables = extract_formula_variables(main if body else formula)

    if where and variables and all(
        v.lower() in existing or _lookup_definition(v) for v in variables
    ):
        return display, where

    where_clause = _build_where_clause(variables, existing)
    if where_clause is None and where:
        return display, where
    return display, where_clause


def render_formula_lines(formula: str, *, bullet: str = "• ", indent: str = "  ") -> list[str]:
    """Render a formula as one or two lines with variable glossary."""
    display, where = annotate_formula_parts(formula)
    lines = [f"{bullet}{display}".rstrip()]
    if where:
        lines.append(f"{indent}where {where}")
    return lines
