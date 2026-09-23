from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def brier_score(confidences: list[float], correct: list[bool]) -> float:
    _validate_inputs(confidences, correct)
    return round(mean((confidence - float(is_correct)) ** 2 for confidence, is_correct in zip(confidences, correct)), 6)


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    bins: int = 10,
) -> dict:
    _validate_inputs(confidences, correct)
    if bins < 1:
        raise ValueError("bins must be positive")

    rows = []
    total = len(confidences)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        in_bin = [
            (confidence, is_correct)
            for confidence, is_correct in zip(confidences, correct)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if not in_bin:
            rows.append({"bin": index, "count": 0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0})
            continue
        avg_confidence = mean(confidence for confidence, _ in in_bin)
        accuracy = mean(float(is_correct) for _, is_correct in in_bin)
        gap = abs(avg_confidence - accuracy)
        ece += len(in_bin) / total * gap
        rows.append(
            {
                "bin": index,
                "count": len(in_bin),
                "confidence": round(avg_confidence, 6),
                "accuracy": round(accuracy, 6),
                "gap": round(gap, 6),
            }
        )
    return {"ece": round(ece, 6), "brier": brier_score(confidences, correct), "bins": rows}


def load_confidence_rows(path: Path) -> tuple[list[float], list[bool]]:
    rows = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else _read_jsonl(path)
    confidences = []
    correct = []
    for row in rows:
        confidences.append(float(row["confidence"]))
        correct.append(bool(row["correct"]))
    return confidences, correct


def _validate_inputs(confidences: list[float], correct: list[bool]) -> None:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        raise ValueError("at least one prediction is required")
    if any(confidence < 0 or confidence > 1 for confidence in confidences):
        raise ValueError("confidence values must be in [0, 1]")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simple calibration metrics for student outputs.")
    parser.add_argument("--input", type=Path, default=Path("reports/confidence_rows.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/calibration_metrics.json"))
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    confidences, correct = load_confidence_rows(args.input)
    report = expected_calibration_error(confidences, correct, bins=args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
