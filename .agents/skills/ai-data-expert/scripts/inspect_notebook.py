#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import nbformat

IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")

def inspect(path: Path) -> dict:
    nb = nbformat.read(path, as_version=4)
    markdown = "\n\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    code = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
    ids = IDENT.findall(markdown)
    target = None
    for line in markdown.splitlines():
        low = line.lower()
        if "target" in low or "예측 목표" in line or "목표변수" in line:
            found = IDENT.findall(line)
            if found:
                target = found[0]
                break
    return {
        "path": str(path.resolve()),
        "cell_count": len(nb.cells),
        "markdown_cell_count": sum(c.cell_type=="markdown" for c in nb.cells),
        "code_cell_count": sum(c.cell_type=="code" for c in nb.cells),
        "empty_code_cells": [i for i,c in enumerate(nb.cells) if c.cell_type=="code" and not c.source.strip()],
        "candidate_identifiers": sorted(set(ids)),
        "target_candidate": target,
        "problem_text": markdown[:12000],
        "existing_code_chars": len(code),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out")
    args=ap.parse_args()
    result=inspect(Path(args.notebook))
    text=json.dumps(result,ensure_ascii=False,indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text,encoding="utf-8")

if __name__=="__main__":
    main()
