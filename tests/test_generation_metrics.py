import json

import pytest

from scripts.generation_metrics import (
    distinct_n,
    generation_pair_metrics,
    load_generation_rows,
    summarize_generation_rows,
    tokenize,
)


def test_tokenize_normalizes_simple_text() -> None:
    assert tokenize("Hello, HELLO world!") == ["hello", "hello", "world"]


def test_distinct_n_tracks_repetition() -> None:
    tokens = ["yes", "yes", "yes"]

    assert distinct_n(tokens, 1) == pytest.approx(1 / 3)


def test_generation_pair_metrics_compare_teacher_and_student() -> None:
    metrics = generation_pair_metrics(
        teacher_text="The model learns from soft labels.",
        student_text="The student learns from labels.",
        reference_text="A student model learns from soft supervision.",
    )

    assert metrics["student_tokens"] == 5
    assert metrics["teacher_student_unigram_f1"] > 0.5
    assert "student_reference_unigram_f1" in metrics


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
