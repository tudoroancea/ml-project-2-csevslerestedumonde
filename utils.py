import torch
import numpy as np


def bce_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Computes the binary cross entropy loss.
    Args:
        pred: A tensor of shape [N, 1, H, W] or [N, H, W] where N is the batch size.
            Each value represents the probability that the corresponding pixel is
            a road.
        target: A tensor of shape [N, 1, H, W] or [N, H, W] where N is the batch size.
            Each value is 1 for pixels that are roads, and 0 for the rest.
    Returns:
        The mean loss for the batch.
    """
    return torch.nn.functional.binary_cross_entropy(
        input=pred, target=target, weight=None, reduction="mean"
    )


def f1_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    """
    Computes the F1 score, also known as balanced F-score or F-measure
    Args:
        pred: A tensor of shape [N, 1, H, W] or [N, H, W] where N is the batch size.
            Each value represents the probability that the corresponding pixel is
            a road.
        target: A tensor of shape [N, 1, H, W] or [N, H, W] where N is the batch size.
            Each value is 1 for pixels that are roads, and 0 for the rest.
        eps: added to the denominator for numerical stability
    Returns:
        The mean F1 score for the batch.
    """
    pred = torch.round(pred)
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return f1


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    """
    Computes the Sørensen–Dice loss.
    Args:
        pred: A tensor of shape [N, 1, H, W] or [N, H, W] where N is the batch size.
            Each value represents the probability that the corresponding pixel is
            a road.
        target: A tensor of shape [N, 1, H, W] or [N, H, W] where N is the batch size.
            Each value is 1 for pixels that are roads, and 0 for the rest.
        eps: added to the denominator for numerical stability
    Returns:
        The mean loss for the batch.
    """
    pred = torch.round(pred)
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return 1 - dice
