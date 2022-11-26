import torch

__all__ = [
    "dice_coeff",
    "jaccard_index",
    "dice_loss",
    "iou_loss",
    "bce_loss",
]

# classification metrics ======================================================
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    """
    Computes the Sorensen-Dice coefficient, also known as the F1 score.
    Args:
        pred: A tensor of shape [N, 2, H, W] where N is the batch size.
            Each value represents the probability that the corresponding pixel is
            a road.
        target: A tensor of shape [N, 2, H, W] where N is the batch size.
            Each value is 1 for pixels that are roads, and 0 for the rest.
        eps: added to the denominator for numerical stability
    Returns:
        The mean F1 score for the batch.
    """
    return torch.mean(
        (eps + 2 * torch.sum(pred[:, 0, :, :] * target[:, 0, :, :], dim=0))
        / (
            eps
            + torch.sum(pred[:, 0, :, :] ** 2, dim=0)
            + torch.sum(target[:, 0, :, :] ** 2, dim=0)
        )
    )


def jaccard_index(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    """
    Computes the Jaccard index, also known as the intersection-over-union.
    Args:
        pred: A tensor of shape [N, 1, H, W] or [N, H, W] where N is the batch size.
            Each value represents the probability that the corresponding pixel is
            a road.
        target: A tensor of shape [N, 1, H, W] or [N, H, W] where N is the batch size.
            Each value is 1 for pixels that are roads, and 0 for the rest.
        eps: added to the denominator for numerical stability
    Returns:
        The mean intersection over union for the batch.
    """
    # pred = torch.round(pred)
    intersection = torch.sum(pred * target, dim=0)
    union = torch.sum(pred, dim=0) + torch.sum(target, dim=0) - intersection
    return torch.mean((intersection + eps) / (union + eps))


# classfication loss  =========================================================
def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    """
    Computes the Dice loss.
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
    return 1 - dice_coeff(pred, target, eps)


def iou_loss(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    """
    Computes the IoU loss.
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
    return 1 - jaccard_index(pred, target, eps)


def bce_loss(pred: torch.Tensor, target: torch.Tensor, road_weight: float = 0.5):
    """
    Computes the binary cross entropy loss.
    Args:
        pred: A tensor of shape [N, 2, H, W] where N is the batch size.
            Each value represents the probability that the corresponding pixel is
            a road.
        target: A tensor of shape [N, 2, H, W] where N is the batch size.
            Each value is 1 for pixels that are roads, and 0 for the rest.
    Returns:
        The mean loss for the batch.
    """
    # return torch.nn.functional.binary_cross_entropy(
    #     input=pred[:, 0, :, :], target=target[:, 0, :, :], weight=None, reduction="sum"
    # )
    return -torch.sum(
        road_weight * target[:, 0, :, :] * torch.log(pred[:, 0, :, :])
        + (1 - road_weight) * (1 - target[:, 0, :, :]) * torch.log(1 - pred[:, 0, :, :])
    )
