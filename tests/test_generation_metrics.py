import json

import pytest

from scripts.generation_metrics import (
    distinct_n,
    generation_pair_metrics,
    generation_quality_flags,
    jaccard_overlap,
    load_generation_rows,
    set_coverage,
    summarize_generation_rows,
    tokenize,
)


def test_tokenize_normalizes_simple_text() -> None:
    assert tokenize("Hello, HELLO world!") == ["hello", "hello", "world"]


def test_distinct_n_tracks_repetition() -> None:
    tokens = ["yes", "yes", "yes"]

    assert distinct_n(tokens, 1) == pytest.approx(1 / 3)


def test_set_coverage_handles_empty_left_side() -> None:
    assert set_coverage([], ["teacher"]) == 1.0
    assert set_coverage(["a", "b"], ["b", "c"]) == 0.5


def test_jaccard_overlap_handles_shared_and_empty_sets() -> None:
    assert jaccard_overlap([], []) == 1.0
    assert jaccard_overlap(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_generation_pair_metrics_compare_teacher_and_student() -> None:
    metrics = generation_pair_metrics(
        teacher_text="The model learns from soft labels.",
        student_text="The student learns from labels.",
        reference_text="A student model learns from soft supervision.",
    )

    assert metrics["student_tokens"] == 5
    assert metrics["student_length_delta"] == -1
    assert metrics["teacher_student_unigram_f1"] > 0.5
    assert metrics["teacher_student_jaccard"] > 0.4
    assert 0 <= metrics["student_novel_unigram_rate"] <= 1
    assert metrics["student_unigram_teacher_coverage"] > 0.5
    assert "student_reference_unigram_f1" in metrics


def test_generation_quality_flags_catch_weak_outputs() -> None:
    flags = generation_quality_flags(
        {
            "teacher_student_unigram_f1": 0.1,
            "length_ratio": 0.2,
            "student_repetition_rate": 0.7,
            "teacher_student_jaccard": 0.95,
        }
    )

    assert flags == [
        "low_teacher_overlap",
        "too_short",
        "repetitive_student",
        "near_teacher_copy",
    ]


def test_summarize_generation_rows_supports_key_aliases() -> None:
    report = summarize_generation_rows(
        [
            {
                "teacher_response": "Large model gives a concise answer.",
                "student_response": "Small model gives a concise answer.",
                "response": "A concise answer from the model.",
            },
            {
                "teacher": "The teacher uses more details.",
                "student": "The student uses details.",
            },
        ]
    )

    assert report["rows"] == 2
    assert report["summary"]["student_tokens"] > 0
    assert "flag_counts" in report
    assert "quality_flags" in report["per_row"][0]


def test_summarize_generation_rows_requires_outputs() -> None:
    with pytest.raises(ValueError, match="missing"):
        summarize_generation_rows([{"teacher": "ok"}])


def test_load_generation_rows_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"teacher": "a b", "student": "a c"}) + "\n",
        encoding="utf-8",
    )

    assert load_generation_rows(path)[0]["teacher"] == "a b"
