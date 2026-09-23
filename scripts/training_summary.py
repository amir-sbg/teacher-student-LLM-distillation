from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_metric_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("history", payload.get("metrics", []))
        if not isinstance(payload, list):
            raise ValueError("JSON metrics must be a list or contain a history list")
        return payload
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_training(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one metric row is required")

    train_losses = _values(rows, "train_loss", "loss")
    val_losses = _values(rows, "eval_loss", "val_loss", "validation_loss")
    soft_losses = _values(rows, "soft_kl", "soft_loss", "kl_loss")
    hard_losses = _values(rows, "hard_loss", "ce_loss")
    best_index, best_val = _best_row(rows, ("eval_loss", "val_loss", "validation_loss"))
    last_train = train_losses[-1] if train_losses else None
    last_val = val_losses[-1] if val_losses else None
    return {
        "steps": len(rows),
        "first_train_loss": train_losses[0] if train_losses else None,
        "last_train_loss": last_train,
        "best_validation_loss": best_val,
        "best_validation_step": rows[best_index].get("step", best_index) if best_index is not None else None,
        "last_validation_loss": last_val,
        "validation_gap": round(last_val - last_train, 6) if last_train is not None and last_val is not None else None,
        "avg_soft_loss": round(mean(soft_losses), 6) if soft_losses else None,
        "avg_hard_loss": round(mean(hard_losses), 6) if hard_losses else None,
    }


def _values(rows: list[dict], *keys: str) -> list[float]:
    values = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
                break
    return values


def _best_row(rows: list[dict], keys: tuple[str, ...]) -> tuple[int | None, float | None]:
    best_index = None
    best_value = None
    for index, row in enumerate(rows):
        values = _values([row], *keys)
        if not values:
            continue
        value = values[0]
        if best_value is None or value < best_value:
            best_index = index
            best_value = value
    return best_index, best_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize student distillation training metrics.")
    parser.add_argument("--input", type=Path, default=Path("reports/training_history.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/training_summary.json"))
    args = parser.parse_args()

    report = summarize_training(load_metric_rows(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
