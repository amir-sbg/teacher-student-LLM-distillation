import json

import pytest

from scripts.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    load_confidence_rows,
)


def test_expected_calibration_error_bins_predictions() -> None:
    report = expected_calibration_error(
        confidences=[0.2, 0.4, 0.8, 1.0],
        correct=[False, True, True, True],
        bins=2,
    )

    assert report["ece"] > 0
    assert report["brier"] == brier_score([0.2, 0.4, 0.8, 1.0], [False, True, True, True])
    assert sum(row["count"] for row in report["bins"]) == 4


def test_calibration_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_calibration_error([1.2], [True])


def test_load_confidence_rows_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "confidence.jsonl"
    path.write_text(
        json.dumps({"confidence": 0.7, "correct": True}) + "\n",
        encoding="utf-8",
    )

    confidences, correct = load_confidence_rows(path)

    assert confidences == [0.7]
    assert correct == [True]
