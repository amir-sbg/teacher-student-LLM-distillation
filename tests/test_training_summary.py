import json

import pytest

from scripts.training_summary import load_metric_rows, summarize_training


def test_training_summary_tracks_best_validation_loss() -> None:
    report = summarize_training(
        [
            {"step": 1, "train_loss": 2.2, "eval_loss": 2.4, "soft_kl": 0.4, "hard_loss": 2.8},
            {"step": 2, "train_loss": 1.8, "eval_loss": 2.0, "soft_kl": 0.3, "hard_loss": 2.4},
            {"step": 3, "train_loss": 1.5, "eval_loss": 2.1, "soft_kl": 0.2, "hard_loss": 2.0},
        ]
    )

    assert report["steps"] == 3
    assert report["best_validation_loss"] == 2.0
    assert report["best_validation_step"] == 2
    assert report["validation_gap"] == pytest.approx(0.6)
    assert report["avg_soft_loss"] == pytest.approx(0.3)
    assert report["soft_to_hard_loss_ratio"] == pytest.approx(0.125)
    assert report["recent_train_slope"] < 0
    assert report["recent_validation_slope"] < 0
    assert report["training_state"] == "improving"


def test_training_summary_flags_diverging_validation() -> None:
    report = summarize_training(
        [
            {"train_loss": 2.0, "eval_loss": 2.0},
            {"train_loss": 1.8, "eval_loss": 2.1},
            {"train_loss": 1.6, "eval_loss": 2.3},
        ]
    )

    assert report["training_state"] == "possible_overfitting"


def test_training_summary_validates_trend_window() -> None:
    with pytest.raises(ValueError, match="trend_window"):
        summarize_training([{"loss": 1.0}], trend_window=1)


def test_load_metric_rows_supports_jsonl(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps({"step": 1, "loss": 2.0}) + "\n" + json.dumps({"step": 2, "loss": 1.8}) + "\n",
        encoding="utf-8",
    )

    assert load_metric_rows(path)[1]["loss"] == 1.8


def test_training_summary_requires_rows() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_training([])
