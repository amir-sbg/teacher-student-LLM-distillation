from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_split(path: Path) -> dict:
    rows = read_jsonl(path)
    if not rows:
        return {
            "rows": 0,
            "avg_tokens": 0.0,
            "avg_supervised_tokens": 0.0,
            "top_k": 0,
            "min_top_k": 0,
            "max_top_k": 0,
            "empty_supervision": 0,
            "missing_teacher_topk": 0,
            "loss_mask_length_mismatches": 0,
        }

    token_lengths = [len(row.get("input_ids", [])) for row in rows]
    supervised = [sum(row.get("loss_mask", [])) for row in rows]
    top_k_values = []
    missing_teacher_topk = 0
    length_mismatches = 0
    for row in rows:
        input_ids = row.get("input_ids", [])
        loss_mask = row.get("loss_mask", [])
        teacher_indices = row.get("teacher_topk_indices", [])
        if loss_mask and len(loss_mask) != len(input_ids):
            length_mismatches += 1
        if sum(loss_mask) > 0 and not teacher_indices:
            missing_teacher_topk += 1
        top_k_values.extend(len(indices) for indices in teacher_indices if indices)
    return {
        "rows": len(rows),
        "avg_tokens": round(mean(token_lengths), 2),
        "avg_supervised_tokens": round(mean(supervised), 2),
        "top_k": int(max(top_k_values)) if top_k_values else 0,
        "min_top_k": int(min(top_k_values)) if top_k_values else 0,
        "max_top_k": int(max(top_k_values)) if top_k_values else 0,
        "empty_supervision": sum(1 for value in supervised if value == 0),
        "missing_teacher_topk": missing_teacher_topk,
        "loss_mask_length_mismatches": length_mismatches,
    }


def audit_cache(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "teacher_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
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
        if stats["missing_teacher_topk"]:
            warnings.append(f"{split} has supervised rows without teacher top-k logits")
        if stats["loss_mask_length_mismatches"]:
            warnings.append(f"{split} has loss masks with a different length than input_ids")
        if stats["min_top_k"] and stats["min_top_k"] != stats["max_top_k"]:
            warnings.append(f"{split} has inconsistent teacher top-k widths")
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
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
