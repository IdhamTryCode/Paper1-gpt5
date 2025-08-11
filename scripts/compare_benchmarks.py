#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import csv


def safe_get(d, path, default=None):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_language(val):
    if val is None:
        return None
    s = str(val).lower()
    if 'python' in s:
        return 'Python'
    if 'rust' in s:
        return 'Rust'
    return val


def collect_rows(root: Path):
    rows = []
    for p in root.rglob('*.json'):
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        row = {
            'file': str(p.relative_to(root)),
            'dataset': data.get('dataset'),
            'model_name': data.get('model_name'),
            'framework': data.get('framework'),
            'language': normalize_language(data.get('language')),
            'training_time_seconds': safe_get(data, 'performance_metrics.training_time_seconds'),
            'accuracy': safe_get(data, 'quality_metrics.accuracy'),
            'loss': safe_get(data, 'quality_metrics.loss'),
            'f1_score': safe_get(data, 'quality_metrics.f1_score'),
            'precision': safe_get(data, 'quality_metrics.precision'),
            'recall': safe_get(data, 'quality_metrics.recall'),
        }
        rows.append(row)
    return rows


def main():
    if len(sys.argv) < 3:
        print('Usage: compare_benchmarks.py <dir1> <dir2> [<out_csv>]')
        sys.exit(1)
    d1 = Path(sys.argv[1]).resolve()
    d2 = Path(sys.argv[2]).resolve()
    out_csv = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else d1 / 'comparison_combined.csv'

    rows = []
    rows.extend(collect_rows(d1))
    rows.extend(collect_rows(d2))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['file'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote comparison CSV: {out_csv}')


if __name__ == '__main__':
    main()


