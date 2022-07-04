import torch
import torch.nn.functional as F


def dice(y, prd_logits, eps=1):
    bs, n_cls, w, h = prd_logits.shape
    p = F.softmax(prd_logits, 1)
    mask = torch.zeros_like(prd_logits)
    mask = mask.scatter_(1, y.view(bs, 1, w, h), 1)
    dice_loss = (2 * (mask * p).sum() + eps) / (mask.sum() + p.sum() + eps)
    return dice_loss


def jaccard_score(gt_mask, prd_logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        gt_mask: a tensor of shape [B, H, W] or [B, 1, H, W].
        prd_logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or prd_logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = prd_logits.shape[1]

    gt_mask_onehot = torch.eye(num_classes)[gt_mask.squeeze(1)]
    gt_mask_onehot = gt_mask_onehot.permute(0, 3, 1, 2).float()
    probas = F.softmax(prd_logits, dim=1)
    gt_mask_onehot = gt_mask_onehot.type(prd_logits.type())
    dims = (0,) + tuple(range(2, gt_mask.ndimension()))
    intersection = torch.sum(probas * gt_mask_onehot, dims)
    cardinality = torch.sum(probas + gt_mask_onehot, dims)
    union = cardinality - intersection
    jacc = (intersection / (union + eps)).mean()
    return jacc


def jaccard_loss(gt_mask, prd_logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    jacc_score = jaccard_score(gt_mask, prd_logits)
    return 1 - jacc_score
