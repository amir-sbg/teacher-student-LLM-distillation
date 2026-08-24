import json

from scripts.audit_teacher_cache import audit_cache, summarize_split


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_summarize_split_reports_cache_shape_issues(tmp_path) -> None:
    path = tmp_path / "train_teacher_topk.jsonl"
    write_jsonl(
        path,
        [
            {
                "input_ids": [1, 2, 3],
                "loss_mask": [0, 1, 1],
                "teacher_topk_indices": [[4, 5], [6, 7]],
            },
            {
                "input_ids": [1, 2, 3],
                "loss_mask": [0, 1],
                "teacher_topk_indices": [[4, 5, 6]],
            },
            {
                "input_ids": [1, 2],
                "loss_mask": [0, 1],
                "teacher_topk_indices": [],
            },
        ],
    )

    report = summarize_split(path)

    assert report["rows"] == 3
    assert report["min_top_k"] == 2
    assert report["max_top_k"] == 3
    assert report["missing_teacher_topk"] == 1
    assert report["loss_mask_length_mismatches"] == 1


def test_audit_cache_warns_about_malformed_splits(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    write_jsonl(
        cache_dir / "train_teacher_topk.jsonl",
        [
            {
                "input_ids": [1, 2],
                "loss_mask": [1],
                "teacher_topk_indices": [[3, 4]],
            }
        ],
    )
    write_jsonl(
        cache_dir / "validation_teacher_topk.jsonl",
        [
            {
                "input_ids": [1, 2],
                "loss_mask": [0, 1],
                "teacher_topk_indices": [],
            }
        ],
    )

    report = audit_cache(cache_dir)

    assert any("loss masks" in warning for warning in report["warnings"])
    assert any("without teacher top-k" in warning for warning in report["warnings"])
