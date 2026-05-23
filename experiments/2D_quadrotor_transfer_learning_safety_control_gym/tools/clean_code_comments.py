from __future__ import annotations

import argparse
import ast
import io
import tokenize
from pathlib import Path
from typing import Iterable, Set


def docstring_lines(tree: ast.AST) -> Set[int]:
    lines: Set[int] = set()

    def add_if_docstring(node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            start = getattr(first, "lineno", None)
            end = getattr(first, "end_lineno", start)
            if start is not None and end is not None:
                lines.update(range(int(start), int(end) + 1))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_if_docstring(node)
    return lines


def remove_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    skip = docstring_lines(tree)
    if not skip:
        return source
    kept = []
    for i, line in enumerate(source.splitlines(), start=1):
        if i not in skip:
            kept.append(line)
    return "\n".join(kept) + "\n"


def remove_comments(source: str) -> str:
    tokens = []
    try:
        tokgen = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokgen:
            if tok.type == tokenize.COMMENT:
                continue
            tokens.append(tok)
        cleaned = tokenize.untokenize(tokens)
    except tokenize.TokenError:
        cleaned = source
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped in {"", "--", "---", "----", "====", "=====", "======"}:
            if lines and lines[-1].strip() == "":
                continue
        if any(marker in stripped for marker in ["ONE-FILE", "LAMBDA SWEEP", "FULL DEPLOYMENT"]):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def clean_file(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    cleaned = remove_comments(remove_docstrings(source))
    path.write_text(cleaned, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.py"))
    for path in files:
        clean_file(path)
    print(f"Cleaned files: {len(files)}")


if __name__ == "__main__":
    main()
