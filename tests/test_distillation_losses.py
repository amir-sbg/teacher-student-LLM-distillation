import pytest
import torch

from scripts.distillation_losses import (
    blended_distillation_loss,
    blended_js_distillation_loss,
    blended_topk_distillation_loss,
    distillation_schedule,
    hard_ce_loss,
    masked_mean,
    soft_target_entropy,
    teacher_rank_metrics,
    temperature_js_loss,
    topk_temperature_kl_loss,
    top1_agreement,
    temperature_kl_loss,
)


def test_masked_mean_uses_only_supervised_tokens() -> None:
    values = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    assert masked_mean(values, mask) == torch.tensor(4.0)


def test_distillation_schedule_ramps_teacher_weight() -> None:
    first = distillation_schedule(
        0,
        total_steps=5,
        start_alpha=0.4,
        end_alpha=0.8,
        start_temperature=4.0,
        end_temperature=2.0,
    )
    last = distillation_schedule(
        4,
        total_steps=5,
        start_alpha=0.4,
        end_alpha=0.8,
        start_temperature=4.0,
        end_temperature=2.0,
    )

    assert first == {"alpha": 0.4, "temperature": 4.0}
    assert last == {"alpha": 0.8, "temperature": 2.0}


def test_distillation_schedule_respects_warmup() -> None:
    warmup = distillation_schedule(1, total_steps=6, warmup_steps=2)
    after_warmup = distillation_schedule(5, total_steps=6, warmup_steps=2)

    assert warmup["alpha"] == 0.45
    assert warmup["temperature"] == 3.0
    assert after_warmup["alpha"] == 0.75


def test_temperature_kl_is_zero_for_matching_logits() -> None:
    logits = torch.tensor([[[1.0, 0.5, -0.5], [0.1, 0.2, 0.3]]])
    mask = torch.tensor([[1, 1]], dtype=torch.bool)

    loss = temperature_kl_loss(logits, logits, temperature=2.0, mask=mask)

    assert loss.item() == pytest.approx(0.0, abs=2e-7)


def test_temperature_js_is_symmetric_and_bounded() -> None:
    student = torch.tensor([[[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]]])
    teacher = torch.tensor([[[0.0, 2.0, -1.0], [0.3, 0.2, 0.1]]])

    forward = temperature_js_loss(student, teacher, temperature=2.0)
    backward = temperature_js_loss(teacher, student, temperature=2.0)

    assert forward.item() > 0.0
    assert forward.item() == pytest.approx(backward.item())
    assert forward.item() < temperature_kl_loss(student, teacher, temperature=2.0).item()


def test_soft_target_entropy_uses_temperature_and_mask() -> None:
    sharp = torch.tensor([[[8.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)

    cold = soft_target_entropy(sharp, temperature=1.0, mask=mask)
    warm = soft_target_entropy(sharp, temperature=4.0, mask=mask)

    assert cold.item() < warm.item()


def test_top1_agreement_tracks_teacher_choices() -> None:
    student = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]]])
    teacher = torch.tensor([[[1.0, 0.0], [4.0, 5.0], [0.0, 2.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    assert top1_agreement(student, teacher, mask=mask).item() == pytest.approx(1.0)


def test_teacher_rank_metrics_track_student_ordering() -> None:
    student = torch.tensor([[[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]])
    teacher = torch.tensor([[[4.0, 0.0, 0.0], [0.0, 0.0, 4.0]]])

    metrics = teacher_rank_metrics(student, teacher, k=2)

    assert metrics["top1_agreement"].item() == pytest.approx(0.5)
    assert metrics["topk_agreement"].item() == pytest.approx(0.5)
    assert metrics["mean_reciprocal_rank"].item() == pytest.approx(2 / 3)
    assert metrics["mean_teacher_token_rank"].item() == pytest.approx(2.0)


def test_teacher_rank_metrics_reject_bad_k() -> None:
    logits = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="k"):
        teacher_rank_metrics(logits, logits, k=0)


def test_hard_ce_ignores_prompt_positions() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ]
    )
    labels = torch.tensor([[-100, 1, 2]])

    assert hard_ce_loss(logits, labels).item() < 0.05


def test_blended_loss_returns_components() -> None:
    student = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    teacher = torch.tensor([[[1.5, 0.0], [0.0, 1.5]]])
    labels = torch.tensor([[-100, 1]])

    report = blended_distillation_loss(
        student,
        teacher,
        labels,
        alpha=0.6,
        temperature=2.0,
    )

    assert set(report) == {"loss", "kl_loss", "ce_loss"}
    assert report["loss"].item() > 0
    assert report["loss"].item() == pytest.approx(
        0.6 * report["kl_loss"].item() + 0.4 * report["ce_loss"].item()
    )


def test_blended_loss_rejects_bad_alpha() -> None:
    logits = torch.zeros(1, 1, 2)
    labels = torch.tensor([[1]])

    with pytest.raises(ValueError, match="alpha"):
        blended_distillation_loss(logits, logits, labels, alpha=1.2)


def test_blended_js_loss_returns_components() -> None:
    student = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    teacher = torch.tensor([[[0.5, 1.0], [0.0, 1.5]]])
    labels = torch.tensor([[-100, 1]])

    report = blended_js_distillation_loss(
        student,
        teacher,
        labels,
        alpha=0.4,
        temperature=2.0,
    )

    assert set(report) == {"loss", "js_loss", "ce_loss"}
    assert report["loss"].item() == pytest.approx(
        0.4 * report["js_loss"].item() + 0.6 * report["ce_loss"].item()
    )


def test_topk_kl_is_zero_when_student_matches_teacher_support() -> None:
    student = torch.tensor([[[2.0, -100.0, -100.0, 1.0], [-100.0, 3.0, 0.0, -100.0]]])
    indices = torch.tensor([[[0, 3], [1, 2]]])
    teacher_logits = torch.gather(student, dim=-1, index=indices)
    mask = torch.tensor([[1, 1]], dtype=torch.bool)

    loss = topk_temperature_kl_loss(
        student,
        indices,
        teacher_logits,
        temperature=2.0,
        mask=mask,
    )

    assert loss.item() == pytest.approx(0.0, abs=2e-7)


def test_topk_kl_penalizes_student_mass_outside_teacher_support() -> None:
    indices = torch.tensor([[[0, 1]]])
    teacher_logits = torch.tensor([[[2.0, 1.0]]])
    aligned = torch.tensor([[[2.0, 1.0, -8.0]]])
    leaked = torch.tensor([[[2.0, 1.0, 8.0]]])

    aligned_loss = topk_temperature_kl_loss(aligned, indices, teacher_logits)
    leaked_loss = topk_temperature_kl_loss(leaked, indices, teacher_logits)

    assert leaked_loss > aligned_loss


def test_topk_blended_loss_returns_components() -> None:
    student = torch.tensor([[[2.0, 0.2, 0.0], [0.1, 1.5, 0.0]]])
    indices = torch.tensor([[[0, 1], [1, 2]]])
    teacher_logits = torch.tensor([[[1.6, 0.2], [1.7, -0.2]]])
    labels = torch.tensor([[-100, 1]])

    report = blended_topk_distillation_loss(
        student,
        indices,
        teacher_logits,
        labels,
        alpha=0.5,
        temperature=2.5,
    )

    assert set(report) == {"loss", "kl_loss", "ce_loss"}
    assert report["loss"].item() == pytest.approx(
        0.5 * report["kl_loss"].item() + 0.5 * report["ce_loss"].item()
    )


def test_topk_kl_rejects_out_of_range_indices() -> None:
    student = torch.zeros(1, 1, 3)
    indices = torch.tensor([[[0, 4]]])
    teacher_logits = torch.zeros(1, 1, 2)

    with pytest.raises(ValueError, match="outside"):
        topk_temperature_kl_loss(student, indices, teacher_logits)
