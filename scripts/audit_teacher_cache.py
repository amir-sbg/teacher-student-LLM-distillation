from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_split(path: Path) -> dict:
    rows = read_jsonl(path)
    if not rows:
        return {"rows": 0, "avg_tokens": 0.0, "avg_supervised_tokens": 0.0, "top_k": 0}

    token_lengths = [len(row.get("input_ids", [])) for row in rows]
    supervised = [sum(row.get("loss_mask", [])) for row in rows]
    top_k_values = [
        len(row.get("teacher_topk_indices", [[]])[0])
        for row in rows
        if row.get("teacher_topk_indices")
    ]
    return {
        "rows": len(rows),
        "avg_tokens": round(mean(token_lengths), 2),
        "avg_supervised_tokens": round(mean(supervised), 2),
        "top_k": int(max(top_k_values)) if top_k_values else 0,
        "empty_supervision": sum(1 for value in supervised if value == 0),
    }


def audit_cache(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "teacher_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    splits = {
        split: summarize_split(cache_dir / f"{split}_teacher_topk.jsonl")
        for split in ("train", "validation")
    }
    warnings = []
    for split, stats in splits.items():
        if stats["rows"] == 0:
            warnings.append(f"{split} cache is empty")
        if stats["empty_supervision"]:
            warnings.append(f"{split} has rows with no supervised response tokens")
    return {"manifest": manifest, "splits": splits, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cached teacher logits before student training.")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/teacher_cache"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = audit_cache(args.cache_dir)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

