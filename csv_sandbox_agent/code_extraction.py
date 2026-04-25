from __future__ import annotations

import ast
import re


class CodeExtractionError(ValueError):
    """Raised when LLM output cannot be reduced to valid Python code."""


_FENCE_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+.-]*)[ \t]*\r?\n(?P<code>.*?)(?:\r?\n)?```",
    re.DOTALL,
)


def extract_python_code(response: str) -> str:
    """Extract valid Python code from brittle LLM output.

    Strategy:
    1. Prefer the largest Python fenced block.
    2. Else prefer the largest generic fenced block.
    3. Else use the full response after stripping obvious prose.
    4. Require ast.parse success before returning.
    """

    if not response or not response.strip():
        raise CodeExtractionError("LLM response is empty")

    blocks = [
        (match.group("lang").strip().lower(), match.group("code").strip())
        for match in _FENCE_RE.finditer(response)
    ]

    python_blocks = [
        code for lang, code in blocks if lang in {"python", "py", "python3"}
    ]
    if python_blocks:
        return _validate(max(python_blocks, key=len))

    generic_blocks = [code for lang, code in blocks if not lang]
    if generic_blocks:
        return _validate(max(generic_blocks, key=len))

    stripped = _strip_obvious_prose(response)
    return _validate(stripped)


def _strip_obvious_prose(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""

    start_markers = (
        "import ",
        "from ",
        "def ",
        "class ",
        "if __name__",
        "#!",
        '"""',
        "'''",
    )
    start = 0
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(start_markers):
            start = idx
            break

    candidate = lines[start:]
    while candidate and candidate[-1].strip().lower() in {
        "hope this helps!",
        "let me know if you need anything else.",
    }:
        candidate.pop()
    return "\n".join(candidate).strip()


def _validate(code: str) -> str:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise CodeExtractionError(
            f"Extracted response is not valid Python: line {exc.lineno}: {exc.msg}"
        ) from exc
    return code.strip() + "\n"
