from __future__ import annotations

import re

_ESCAPED_MUL_RE = re.compile(r"\\\*")
_ESCAPED_ELLIPSIS_RE = re.compile(r"\\\.{3,}")
_UNNECESSARY_ESCAPE_RE = re.compile(r"\\([()\[\]{}_.,;:!?+\-=|/])")
_SQRT_BRACE_RE = re.compile(r"\\sqrt\{([^}]+)\}")
_SQRT_PAREN_RE = re.compile(r"\\sqrt\(([^)]+)\)")
_FORMULA_ELLIPSIS_RE = re.compile(r"(?<=[×x*\d)])\s*\.\.\.\s*(?=[×x*\d(])")


_LATEX_FRAC_RE = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")
_LATEX_CMD_RE = re.compile(r"\\(?:bar|hat|overline)\{([^}]*)\}")
_LATEX_BAR_RE = re.compile(r"\\bar\{([^}]*)\}")
_LATEX_SUM_RE = re.compile(r"\\sum\b")


def sanitize_plaintext(text: str) -> str:
    """Remove markdown/JSON/LaTeX escape artifacts for plain .txt output."""
    if not text:
        return text

    text = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", text)
    text = _LATEX_CMD_RE.sub(r"\1", text)
    text = _LATEX_BAR_RE.sub(r"\1", text)
    text = _LATEX_SUM_RE.sub("Σ", text)
    text = _ESCAPED_MUL_RE.sub("×", text)
    text = _ESCAPED_ELLIPSIS_RE.sub("…", text)
    text = _SQRT_BRACE_RE.sub(r"√(\1)", text)
    text = _SQRT_PAREN_RE.sub(r"√(\1)", text)
    text = _UNNECESSARY_ESCAPE_RE.sub(r"\1", text)
    text = _FORMULA_ELLIPSIS_RE.sub(" … ", text)
    return re.sub(r"  +", " ", text)
