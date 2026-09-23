from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_schedule(
    step: int,
    total_steps: int,
    *,
    start_alpha: float = 0.45,
    end_alpha: float = 0.75,
    start_temperature: float = 3.0,
    end_temperature: float = 2.0,
    warmup_steps: int = 0,
) -> dict[str, float]:
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must not be negative")
    if not 0 <= start_alpha <= 1 or not 0 <= end_alpha <= 1:
        raise ValueError("alpha values must be between 0 and 1")
    if start_temperature <= 0 or end_temperature <= 0:
        raise ValueError("temperature values must be positive")

    clipped_step = min(step, total_steps - 1)
    if clipped_step < warmup_steps:
        progress = 0.0
    else:
        denominator = max(total_steps - warmup_steps - 1, 1)
        progress = min((clipped_step - warmup_steps) / denominator, 1.0)
    return {
        "alpha": _lerp(start_alpha, end_alpha, progress),
        "temperature": _lerp(start_temperature, end_temperature, progress),
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if mask.shape != values.shape:
        raise ValueError("mask must have the same shape as values")
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


def temperature_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocab)")

    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits / temperature
    token_kl = F.kl_div(
        F.log_softmax(scaled_student, dim=-1),
        F.softmax(scaled_teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return masked_mean(token_kl, mask) * temperature**2


def temperature_js_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocab)")

    student_probabilities = F.softmax(student_logits / temperature, dim=-1)
    teacher_probabilities = F.softmax(teacher_logits / temperature, dim=-1)
    mixture = 0.5 * (student_probabilities + teacher_probabilities)

    student_log = torch.log(student_probabilities.clamp_min(1e-12))
    teacher_log = torch.log(teacher_probabilities.clamp_min(1e-12))
    mixture_log = torch.log(mixture.clamp_min(1e-12))
    token_js = 0.5 * (
        teacher_probabilities * (teacher_log - mixture_log)
    ).sum(dim=-1) + 0.5 * (
        student_probabilities * (student_log - mixture_log)
    ).sum(dim=-1)
    return masked_mean(token_js, mask) * temperature**2


def soft_target_entropy(
    logits: torch.Tensor,
    temperature: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocab)")

    probabilities = F.softmax(logits / temperature, dim=-1)
    token_entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
    return masked_mean(token_entropy, mask)


def top1_agreement(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocab)")

    agreement = student_logits.argmax(dim=-1).eq(teacher_logits.argmax(dim=-1)).float()
    return masked_mean(agreement, mask)


def topk_temperature_kl_loss(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape (batch, sequence, vocab)")
    if teacher_topk_indices.shape != teacher_topk_logits.shape:
        raise ValueError("teacher top-k indices and logits must have the same shape")
    if teacher_topk_indices.ndim != 3:
        raise ValueError("teacher top-k tensors must have shape (batch, sequence, top_k)")
    if teacher_topk_indices.shape[:2] != student_logits.shape[:2]:
        raise ValueError("teacher top-k tensors must match the student batch and sequence shape")
    if teacher_topk_indices.numel() == 0:
        raise ValueError("teacher top-k tensors must not be empty")
    if teacher_topk_indices.min().item() < 0 or teacher_topk_indices.max().item() >= student_logits.shape[-1]:
        raise ValueError("teacher_topk_indices contain ids outside the student vocabulary")

    gathered_student = torch.gather(student_logits, dim=-1, index=teacher_topk_indices.long())
    teacher_distribution = F.softmax(teacher_topk_logits / temperature, dim=-1)
    student_log_distribution = F.log_softmax(gathered_student / temperature, dim=-1)
    teacher_log_distribution = torch.log(teacher_distribution.clamp_min(1e-12))
    token_kl = (teacher_distribution * (teacher_log_distribution - student_log_distribution)).sum(dim=-1)
    return masked_mean(token_kl, mask) * temperature**2


def hard_ce_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape (batch, sequence, vocab)")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("labels must have shape (batch, sequence)")
    return F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def blended_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if mask is None:
        mask = labels.ne(ignore_index)

    kl = temperature_kl_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
        mask=mask,
    )
    ce = hard_ce_loss(student_logits, labels, ignore_index=ignore_index)
    loss = alpha * kl + (1.0 - alpha) * ce
    return {
        "loss": loss,
        "kl_loss": kl,
        "ce_loss": ce,
    }


def blended_js_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if mask is None:
        mask = labels.ne(ignore_index)

    js = temperature_js_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
        mask=mask,
    )
    ce = hard_ce_loss(student_logits, labels, ignore_index=ignore_index)
    return {
        "loss": alpha * js + (1.0 - alpha) * ce,
        "js_loss": js,
        "ce_loss": ce,
    }


def blended_topk_distillation_loss(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if mask is None:
        mask = labels.ne(ignore_index)

    kl = topk_temperature_kl_loss(
        student_logits,
        teacher_topk_indices,
        teacher_topk_logits,
        temperature=temperature,
        mask=mask,
    )
    ce = hard_ce_loss(student_logits, labels, ignore_index=ignore_index)
    return {
        "loss": alpha * kl + (1.0 - alpha) * ce,
        "kl_loss": kl,
        "ce_loss": ce,
    }


def _lerp(start: float, end: float, weight: float) -> float:
    return float(start + (end - start) * weight)
