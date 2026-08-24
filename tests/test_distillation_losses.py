import pytest
import torch

from scripts.distillation_losses import (
    blended_distillation_loss,
    hard_ce_loss,
    masked_mean,
    temperature_kl_loss,
)


def test_masked_mean_uses_only_supervised_tokens() -> None:
    values = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    assert masked_mean(values, mask) == torch.tensor(4.0)


def test_temperature_kl_is_zero_for_matching_logits() -> None:
    logits = torch.tensor([[[1.0, 0.5, -0.5], [0.1, 0.2, 0.3]]])
    mask = torch.tensor([[1, 1]], dtype=torch.bool)

    loss = temperature_kl_loss(logits, logits, temperature=2.0, mask=mask)

    assert loss.item() == pytest.approx(0.0, abs=2e-7)


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
