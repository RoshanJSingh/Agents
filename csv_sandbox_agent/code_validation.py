from __future__ import annotations

import ast
from pathlib import PurePosixPath


class CodeValidationError(ValueError):
    """Raised when generated Python fails deterministic safety validation."""


FORBIDDEN_IMPORT_ROOTS = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "shutil",
    "multiprocessing",
    "threading",
    "asyncio",
}

ALLOWED_IMPORT_ROOTS = {
    "argparse",
    "datetime",
    "json",
    "joblib",
    "math",
    "numpy",
    "pandas",
    "pathlib",
    "re",
    "sklearn",
    "traceback",
    "typing",
    "warnings",
}

FORBIDDEN_BUILTIN_CALLS = {
    "eval",
    "exec",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
}


def validate_generated_code(code: str, *, max_bytes: int = 100_000) -> None:
    encoded_size = len(code.encode("utf-8"))
    if encoded_size > max_bytes:
        raise CodeValidationError(
            f"Generated script is too large: {encoded_size} bytes > {max_bytes}"
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(
            f"Generated script is not valid Python: line {exc.lineno}: {exc.msg}"
        ) from exc

    visitor = _SafetyVisitor()
    visitor.visit(tree)
    if visitor.errors:
        raise CodeValidationError("; ".join(visitor.errors))

    _validate_contract(code, tree)


def _validate_contract(code: str, tree: ast.AST) -> None:
    required_text = ["--input", "--output-dir", "cleaned_data.csv", "metrics.json"]
    missing = [item for item in required_text if item not in code]
    if missing:
        raise CodeValidationError(
            "Generated script is missing required contract text: " + ", ".join(missing)
        )
    if "argparse" not in code:
        raise CodeValidationError("Generated script must use argparse")

    has_main_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "main":
                has_main_call = True
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "app"
            ):
                has_main_call = True
    if not has_main_call:
        raise CodeValidationError(
            "Generated script must contain an argparse entrypoint or main() call"
        )


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._check_import(node.module, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in FORBIDDEN_BUILTIN_CALLS:
            self.errors.append(f"Forbidden builtin call {name} at line {node.lineno}")
        if name and (
            name.startswith("os.")
            or name.startswith("subprocess.")
            or name.startswith("socket.")
            or name == "shutil.rmtree"
        ):
            self.errors.append(f"Forbidden call {name} at line {node.lineno}")
        if name in {"Path.home", "pathlib.Path.home"}:
            self.errors.append(f"Forbidden Path.home() at line {node.lineno}")
        if name in {"Path", "pathlib.Path"}:
            self._check_path_constructor(node)
        if name == "open":
            self._check_open_call(node)
        self.generic_visit(node)

    def _check_import(self, name: str, lineno: int) -> None:
        root = name.split(".", 1)[0]
        if root in FORBIDDEN_IMPORT_ROOTS:
            self.errors.append(f"Forbidden import {name} at line {lineno}")
        elif root not in ALLOWED_IMPORT_ROOTS:
            self.errors.append(f"Import {name} is not in the allowed import list at line {lineno}")

    def _check_path_constructor(self, node: ast.Call) -> None:
        if not node.args:
            return
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            value = first.value
            if value == "/":
                self.errors.append(f"Forbidden Path('/') at line {node.lineno}")
            elif _is_forbidden_absolute_path(value):
                self.errors.append(
                    f"Forbidden absolute path outside /workspace at line {node.lineno}: {value}"
                )

    def _check_open_call(self, node: ast.Call) -> None:
        if not node.args:
            return
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            value = first.value
            if _is_forbidden_absolute_path(value):
                self.errors.append(
                    f"Forbidden open() absolute path outside /workspace at line {node.lineno}: {value}"
                )


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        if parent:
            return f"{parent}.{func.attr}"
        return func.attr
    return None


def _is_forbidden_absolute_path(value: str) -> bool:
    path = PurePosixPath(value)
    if not path.is_absolute():
        return False
    if ".." in path.parts:
        return True
    return not (value == "/workspace" or value.startswith("/workspace/"))
