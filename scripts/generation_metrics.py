from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


TEACHER_KEYS = ("teacher", "teacher_text", "teacher_output", "teacher_response", "teacher_generation")
STUDENT_KEYS = ("student", "student_text", "student_output", "student_response", "student_generation")
REFERENCE_KEYS = ("reference", "target", "response", "expected_response")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def ngram_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be at least 1")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def overlap_f1(left: Iterable, right: Iterable) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    precision = overlap / len(left_set)
    recall = overlap / len(right_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def set_coverage(left: Iterable, right: Iterable) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set:
        return 1.0
    return len(left_set & right_set) / len(left_set)


def distinct_n(tokens: list[str], n: int) -> float:
    grams = ngram_set(tokens, n)
    total = max(len(tokens) - n + 1, 0)
    if total == 0:
        return 0.0
    return len(grams) / total


def repetition_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def generation_pair_metrics(
    teacher_text: str,
    student_text: str,
    reference_text: Optional[str] = None,
) -> dict[str, float | int]:
    teacher_tokens = tokenize(teacher_text)
    student_tokens = tokenize(student_text)
    student_unigrams = ngram_set(student_tokens, 1)
    teacher_unigrams = ngram_set(teacher_tokens, 1)
    report: dict[str, float | int] = {
        "teacher_tokens": len(teacher_tokens),
        "student_tokens": len(student_tokens),
        "student_length_delta": len(student_tokens) - len(teacher_tokens),
        "length_ratio": len(student_tokens) / max(len(teacher_tokens), 1),
        "teacher_student_unigram_f1": overlap_f1(student_unigrams, teacher_unigrams),
        "teacher_student_bigram_f1": overlap_f1(
            ngram_set(student_tokens, 2),
            ngram_set(teacher_tokens, 2),
        ),
        "student_unigram_teacher_coverage": set_coverage(
            student_unigrams,
            teacher_unigrams,
        ),
        "teacher_unigram_student_coverage": set_coverage(
            teacher_unigrams,
            student_unigrams,
        ),
        "student_novel_unigram_rate": 1.0
        - set_coverage(student_unigrams, teacher_unigrams),
        "student_distinct_1": distinct_n(student_tokens, 1),
        "student_distinct_2": distinct_n(student_tokens, 2),
        "student_repetition_rate": repetition_rate(student_tokens),
        "teacher_repetition_rate": repetition_rate(teacher_tokens),
    }
    if reference_text is not None:
        reference_tokens = tokenize(reference_text)
        report["student_reference_unigram_f1"] = overlap_f1(
            student_unigrams,
            ngram_set(reference_tokens, 1),
        )
        report["teacher_reference_unigram_f1"] = overlap_f1(
            teacher_unigrams,
            ngram_set(reference_tokens, 1),
        )
    return report


def generation_quality_flags(
    metrics: dict[str, float | int],
    *,
    min_unigram_f1: float = 0.30,
    min_length_ratio: float = 0.40,
    max_length_ratio: float = 1.80,
    max_repetition_rate: float = 0.35,
) -> list[str]:
    if min_unigram_f1 < 0:
        raise ValueError("min_unigram_f1 must not be negative")
    if min_length_ratio <= 0 or max_length_ratio < min_length_ratio:
        raise ValueError("length-ratio bounds are inconsistent")
    if not 0 <= max_repetition_rate <= 1:
        raise ValueError("max_repetition_rate must be between 0 and 1")

    flags = []
    if float(metrics["teacher_student_unigram_f1"]) < min_unigram_f1:
        flags.append("low_teacher_overlap")
    if float(metrics["length_ratio"]) < min_length_ratio:
        flags.append("too_short")
    if float(metrics["length_ratio"]) > max_length_ratio:
        flags.append("too_long")
    if float(metrics["student_repetition_rate"]) > max_repetition_rate:
        flags.append("repetitive_student")
    return flags


def summarize_generation_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one generation row is required")

    per_row = []
    for index, row in enumerate(rows):
        teacher_text = _first_text(row, TEACHER_KEYS)
        student_text = _first_text(row, STUDENT_KEYS)
        reference_text = _first_text(row, REFERENCE_KEYS, required=False)
        metrics = generation_pair_metrics(teacher_text, student_text, reference_text)
        per_row.append(
            {
                "index": index,
                **metrics,
                "quality_flags": generation_quality_flags(metrics),
            }
        )

    numeric_keys = sorted(
        {
            key
            for row in per_row
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "index"
        }
    )
    summary = {
        key: float(mean(float(row[key]) for row in per_row if key in row))
        for key in numeric_keys
    }
    flag_counts: dict[str, int] = {}
    for row in per_row:
        for flag in row["quality_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return {
        "rows": len(rows),
        "summary": summary,
        "flag_counts": flag_counts,
        "per_row": per_row,
    }


def load_generation_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON input must contain a list of rows")
        return payload
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _first_text(row: dict, keys: tuple[str, ...], required: bool = True) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if required:
        raise ValueError(f"row is missing one of these text fields: {keys}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize teacher/student generation comparisons.")
    parser.add_argument("--input", type=Path, default=Path("reports/generation_comparisons.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/generation_metrics.json"))
    args = parser.parse_args()

    report = summarize_generation_rows(load_generation_rows(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
