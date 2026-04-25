from __future__ import annotations

import re


KEY_PATTERNS = (
    "Traceback",
    "Error",
    "Exception",
    "KeyError",
    "ValueError",
    "TypeError",
    "ParserError",
    "MemoryError",
    "FileNotFoundError",
    "line",
    "generated_script.py",
)


def compact_error(stdout: str, stderr: str, *, max_chars: int = 6000) -> str:
    primary = stderr if stderr.strip() else stdout
    if not primary.strip():
        return ""

    lines = primary.splitlines()
    tail = lines[-80:]
    generated_frame = _last_line_containing(lines, "generated_script.py")
    final_exception = _final_exception_line(lines)

    selected: list[str] = []
    for line in tail:
        if _is_noisy_internal(line):
            continue
        if any(pattern in line for pattern in KEY_PATTERNS):
            selected.append(line)

    if not selected:
        selected = tail

    if generated_frame and generated_frame not in selected:
        selected.append(generated_frame)
    if final_exception and final_exception not in selected:
        selected.append(final_exception)

    compacted = "\n".join(selected).strip()
    if len(compacted) > max_chars:
        compacted = compacted[-max_chars:]
        first_newline = compacted.find("\n")
        if first_newline != -1:
            compacted = compacted[first_newline + 1 :]
    return compacted.strip()


def failure_fingerprint(
    *,
    stdout: str = "",
    stderr: str = "",
    compacted_error: str = "",
    exit_code: int | None = None,
    timed_out: bool = False,
) -> str:
    text = compacted_error or compact_error(stdout, stderr)
    final_line = _final_exception_line(text.splitlines()) or _last_nonempty(text)
    exception_type = _exception_type(final_line)
    user_line = _generated_script_line(text)
    normalized = _normalize_message(final_line)
    return "|".join(
        [
            f"exc={exception_type or 'unknown'}",
            f"line={user_line or 'unknown'}",
            f"msg={normalized or 'empty'}",
            f"exit={exit_code if exit_code is not None else 'unknown'}",
            f"timeout={timed_out}",
        ]
    )


def _is_noisy_internal(line: str) -> bool:
    lowered = line.lower()
    return "site-packages" in lowered and any(
        pkg in lowered for pkg in ("pandas", "numpy", "sklearn", "scipy")
    )


def _last_line_containing(lines: list[str], needle: str) -> str | None:
    for line in reversed(lines):
        if needle in line:
            return line
    return None


def _final_exception_line(lines: list[str]) -> str | None:
    for line in reversed(lines):
        stripped = line.strip()
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Warning)?:", stripped):
            return stripped
        if re.match(r"^(KeyError|ValueError|TypeError|ParserError|MemoryError|FileNotFoundError):", stripped):
            return stripped
    return None


def _last_nonempty(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _exception_type(line: str | None) -> str | None:
    if not line:
        return None
    match = re.match(r"^([A-Za-z_][\w.]*?(?:Error|Exception|Warning)?):", line.strip())
    if match:
        return match.group(1)
    return None


def _generated_script_line(text: str) -> str | None:
    matches = re.findall(r'generated_script\.py", line (\d+)', text)
    if matches:
        return matches[-1]
    matches = re.findall(r"generated_script\.py.*?line (\d+)", text)
    if matches:
        return matches[-1]
    return None


def _normalize_message(line: str | None) -> str:
    if not line:
        return ""
    normalized = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
    normalized = re.sub(r"/workspace/[^\s'\"]+", "/workspace/PATH", normalized)
    normalized = re.sub(r"\b\d{4,}\b", "N", normalized)
    return normalized.strip()
