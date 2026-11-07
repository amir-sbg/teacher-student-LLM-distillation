import pytest
import torch

from scripts.distillation_losses import (
    adaptive_temperature_kl_loss,
    blended_adaptive_distillation_loss,
    blended_confidence_distillation_loss,
    blended_distillation_loss,
    blended_js_distillation_loss,
    blended_standardized_distillation_loss,
    blended_topk_distillation_loss,
    confidence_weighted_kl_loss,
    distillation_schedule,
    hard_ce_loss,
    linear_cka_similarity,
    margin_weighted_kl_loss,
    masked_mean,
    relational_similarity_loss,
    representation_alignment_loss,
    soft_target_entropy,
    standardized_logit_kl_loss,
    teacher_token_confidence,
    teacher_top2_margin,
    teacher_rank_metrics,
    teacher_adaptive_temperature,
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


def test_standardized_kl_ignores_logit_scale_and_offset() -> None:
    teacher = torch.tensor([[[2.0, 0.5, -1.0], [0.2, -0.4, 1.2]]])
    student = teacher * 4.0 + 7.0

    standardized = standardized_logit_kl_loss(student, teacher, temperature=1.0)
    ordinary = temperature_kl_loss(student, teacher, temperature=1.0)

    assert standardized.item() == pytest.approx(0.0, abs=2e-6)
    assert ordinary > standardized


def test_standardized_blend_backpropagates_through_student() -> None:
    student = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    teacher = torch.tensor([[[1.0, 0.5], [0.2, 1.5]]])
    labels = torch.tensor([[-100, 1]])

    report = blended_standardized_distillation_loss(student, teacher, labels, alpha=0.6)
    report["loss"].backward()

    assert set(report) == {"loss", "kl_loss", "ce_loss"}
    assert student.grad is not None


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


def test_teacher_confidence_tracks_distribution_sharpness() -> None:
    logits = torch.tensor([[[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    confidence = teacher_token_confidence(logits)

    assert confidence[0, 0] > 0.9
    assert confidence[0, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_adaptive_temperature_softens_confident_tokens_more() -> None:
    logits = torch.tensor([[[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    temperature = teacher_adaptive_temperature(logits, 1.0, 4.0)

    assert temperature[0, 0] > 3.5
    assert temperature[0, 1].item() == pytest.approx(1.0, abs=1e-6)


def test_adaptive_temperature_distillation_backpropagates() -> None:
    teacher = torch.tensor([[[5.0, 0.0], [0.2, 0.1]]])
    student = torch.tensor([[[0.0, 4.0], [0.0, 1.0]]], requires_grad=True)
    labels = torch.tensor([[-100, 1]])

    report = blended_adaptive_distillation_loss(student, teacher, labels, alpha=0.6)
    report["loss"].backward()

    assert adaptive_temperature_kl_loss(student, teacher).item() > 0
    assert student.grad is not None


def test_teacher_margin_separates_decisive_and_ambiguous_tokens() -> None:
    logits = torch.tensor([[[8.0, 0.0, 0.0], [1.0, 0.9, 0.0]]])

    margin = teacher_top2_margin(logits)

    assert margin[0, 0] > 0.9
    assert margin[0, 1] < 0.1


def test_margin_weighted_kl_remains_trainable() -> None:
    teacher = torch.tensor([[[5.0, 0.0], [1.0, 0.9]]])
    student = torch.tensor([[[0.0, 4.0], [0.0, 2.0]]], requires_grad=True)

    loss = margin_weighted_kl_loss(student, teacher, temperature=1.0)
    loss.backward()

    assert loss.item() > 0
    assert student.grad is not None


def test_confidence_weighted_kl_downweights_uncertain_teacher_tokens() -> None:
    teacher = torch.tensor([[[8.0, 0.0], [0.0, 0.0]]])
    student = torch.tensor([[[0.0, 8.0], [8.0, 0.0]]])

    weighted = confidence_weighted_kl_loss(
        student,
        teacher,
        temperature=1.0,
        minimum_weight=0.0,
    )
    unweighted = temperature_kl_loss(student, teacher, temperature=1.0)

    assert weighted > unweighted


def test_confidence_blend_returns_trainable_components() -> None:
    student = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    teacher = torch.tensor([[[1.5, 0.0], [0.0, 1.5]]])
    labels = torch.tensor([[-100, 1]])

    report = blended_confidence_distillation_loss(student, teacher, labels, alpha=0.6)
    report["loss"].backward()

    assert set(report) == {"loss", "kl_loss", "ce_loss"}
    assert student.grad is not None


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


def test_linear_cka_is_invariant_to_global_rescaling() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 3.0]]]
    )

    similarity = linear_cka_similarity(hidden, hidden * 4.0)

    assert similarity.item() == pytest.approx(1.0, abs=1e-6)
    assert representation_alignment_loss(hidden, hidden * 4.0).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_linear_cka_supports_different_hidden_widths_and_masks() -> None:
    student = torch.tensor([[[1.0], [2.0], [3.0]]])
    teacher = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [99.0, -99.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    similarity = linear_cka_similarity(student, teacher, mask=mask)

    assert similarity.item() == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_needs_two_active_tokens() -> None:
    hidden = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="at least two"):
        linear_cka_similarity(hidden, hidden, mask=torch.tensor([[1, 0]]))


def test_relational_loss_matches_geometry_across_hidden_widths() -> None:
    student = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    teacher = torch.cat((student * 3.0, torch.zeros(1, 3, 1)), dim=-1)

    loss = relational_similarity_loss(student, teacher)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_relational_loss_respects_mask_and_detects_changed_relations() -> None:
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])
    aligned_teacher = torch.tensor([[[2.0, 0.0], [0.0, 4.0], [-9.0, 9.0]]])
    changed_teacher = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-9.0, 9.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    aligned = relational_similarity_loss(student, aligned_teacher, mask=mask)
    changed = relational_similarity_loss(student, changed_teacher, mask=mask)

    assert aligned.item() == pytest.approx(0.0, abs=1e-7)
    assert changed > aligned


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
