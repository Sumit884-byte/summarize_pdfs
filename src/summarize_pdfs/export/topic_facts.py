from __future__ import annotations

# Short factual statements per statistics branch — not formulas or definitions.
# Used as fallback when LLM output omits branch-level facts.
CANONICAL_TOPIC_FACTS: dict[str, list[str]] = {
    "Descriptive Statistics": [
        "Mean is sensitive to outliers; median is robust to them",
        "Sample statistics estimate population parameters",
        "Standard deviation measures spread around the mean",
        "IQR describes the spread of the middle 50% of data",
    ],
    "Probability & Conditional Probability": [
        "Probability always ranges from 0 to 1",
        "Complement rule always applies: P(not A) = 1 - P(A)",
        "Independent events: P(A and B) = P(A) × P(B)",
        "Mutually exclusive events: P(A or B) = P(A) + P(B)",
        "Conditional probability restricts the sample space to the given event",
    ],
    "Combinatorics & Counting": [
        "Order matters → use permutations; order does not matter → use combinations",
        "With replacement allows reuse; without replacement does not",
        "Factorial n! counts ordered arrangements of n distinct items",
        "Multiplication rule applies when events occur in sequence",
    ],
    "Correlation & Association": [
        "Correlation measures linear association strength and direction, not causation",
        "Pearson r ranges from -1 to +1",
        "Scatter plots reveal association before computing r",
        "Contingency tables summarize association between categorical variables",
    ],
    "Data Types & Study Design": [
        "Nominal → names/categories; ordinal → ordered categories",
        "Interval and ratio scales support meaningful differences; ratio has a true zero",
        "Cross-sectional studies observe at one time; time series track over time",
        "Random sampling reduces selection bias",
    ],
    "Frequency & Distribution": [
        "Relative frequency = count / total; cumulative frequency adds up to a value",
        "Histograms show distribution shape for numeric data",
        "Quartiles divide ordered data into four equal parts",
        "Outliers often fall below Q1 - 1.5×IQR or above Q3 + 1.5×IQR",
    ],
    "Transformations of Data": [
        "Adding a constant shifts mean but not standard deviation",
        "Multiplying by a constant scales both mean and standard deviation",
        "Linear transform: new mean = c × old mean + b; new SD = |c| × old SD",
    ],
    "Exam Skills (MCQ / MSQ / SA)": [
        '"At least one" problems often use the complement rule',
        "MSQ items may require checking independence vs replacement",
        "Eliminate answer choices by scale or data-type rules before calculating",
    ],
}


def canonical_facts_for_topic(topic: str) -> list[str]:
    """Return canonical branch facts for a topic (empty if none)."""
    return list(CANONICAL_TOPIC_FACTS.get(topic, ()))
