#!/usr/bin/env python3
"""
Assemble Rust vs Python benchmark results into a single JSON file
compatible with scripts/perform_statistical_analysis.py.

Usage:
  python scripts/assemble_comparison_results.py <python_dir> <rust_dir> <out_json>
"""

import sys
import json
from pathlib import Path


def normalize_task_key(task_type_value: str) -> str:
    if not task_type_value:
        return "unknown"
    s = str(task_type_value).lower()
    if "deep" in s:
        return "deep_learning"
    if "classical" in s:
        return "classical_ml"
    if "reinforcement" in s or "rl" in s:
        return "reinforcement_learning"
    if "llm" in s or "transformer" in s:
        return "llm"
    return s.replace(" ", "_")


def collect_results(root: Path):
    items = []
    for p in root.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(data)
    return items


def main():
    if len(sys.argv) < 4:
        print("Usage: assemble_comparison_results.py <python_dir> <rust_dir> <out_json>")
        sys.exit(1)

    py_dir = Path(sys.argv[1]).resolve()
    rs_dir = Path(sys.argv[2]).resolve()
    out_path = Path(sys.argv[3]).resolve()

    python_results = collect_results(py_dir)
    rust_results = collect_results(rs_dir)

    assembled = {}

    for side, items in (("python", python_results), ("rust", rust_results)):
        for item in items:
            task_key = normalize_task_key(item.get("task_type"))
            if task_key not in assembled:
                assembled[task_key] = {"python": [], "rust": []}
            assembled[task_key][side].append(item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(assembled, indent=2), encoding="utf-8")
    print(f"Wrote assembled comparison JSON: {out_path}")


if __name__ == "__main__":
    main()


