from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np


def cross_entropy(prd, tgt, weights=None):

    if prd.dim() > 2:
        N, C, H, W = prd.shape
        prd = prd.view(N, C, -1)
        prd = prd.transpose(1, 2)
        prd = prd.contiguous().view(-1, C)

    tgt = tgt.view(-1, 1)

    logp = F.log_softmax(prd)
    logp = logp.gather(1, tgt)
    logp = logp.view(-1)

    if weights is not None:
        weights = weights.gather(0, tgt.view(-1))
        logp = weights * logp
        loss = logp.sum() / weights.sum()
    else:
        loss = logp.mean()

    return -loss


def focal_loss(prd, tgt, gamma=2, weights=None):

    if prd.dim() > 2:
        N, C, H, W = prd.shape
        prd = prd.view(N, C, -1)
        prd = prd.transpose(1, 2)
        prd = prd.contiguous().view(-1, C)

    tgt = tgt.view(-1, 1)

    logp = F.log_softmax(prd)
    logp = logp.gather(1, tgt)
    logp = logp.view(-1)
    p = logp.exp()

    loss = (1 - p) ** gamma * logp

    if weights is not None:
        weights = weights.gather(0, tgt.view(-1))
        loss = weights * loss
        loss = loss.sum() / weights.sum()
    else:
        loss = loss.mean()

    return -loss


def dice(y, prd_logits, eps=1):
    bs, n_cls, w, h = prd_logits.shape
    p = F.softmax(prd_logits, 1)
    mask = torch.zeros_like(prd_logits)
    mask = mask.scatter_(1, y.view(bs, 1, w, h), 1)
    dice_loss = (2 * (mask * p).sum() + eps) / (mask.sum() + p.sum() + eps)
    return dice_loss


def mask_to_class(mask, n_classes=3):
    """Spit masks for `n_classes` binary masks."""
    bs, w, h = mask.shape
    class_mask = np.zeros((n_classes, bs, w, h))
    for c in range(n_classes):
        idxs = mask == c
        class_mask[c, idxs] = 1
    return class_mask


def jaccard(gt_mask, prd_mask, smooth=0.1, reduce="mean"):
    """Calculate Jaccard score per class mean per batch."""
    # prd_mask = prd_mask.argmax(dim=1)

    prd_mask = prd_mask.detach().cpu().numpy()
    gt_mask = gt_mask.squeeze(1).detach().cpu().numpy()

    cm_prd = mask_to_class(prd_mask)
    cm_gt = mask_to_class(gt_mask)

    intersection = np.logical_and(cm_prd, cm_gt).sum(axis=(2, 3))
    union = np.logical_or(cm_prd, cm_gt).sum(axis=(2, 3))

    score = (intersection + smooth) / (union + smooth)
    assert score.max() <= 1
    score = score.mean(axis=1)

    if reduce == "mean":
        return score.mean()
    
    elif reduce == "sum":
        return score.sum()

    else:
        return score



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


def score_model(model, val_dataloader, device, postproc=None):
    """Evaluates metric mean and std per class."""
    scores = []
    val_it = iter(val_dataloader)
    for _ in tqdm(range(len(val_it))):
        img, mask, _ = next(val_it)
        img, mask = img.to(device), mask.to(device)
        prd = model(img.to(device))
        prd = prd.argmax(dim=1).unsqueeze(0)
        prd = F.interpolate(
            prd.type(torch.float32),
            (img.shape[2], img.shape[3]),
            mode="nearest",
        )

        if postproc is not None:
            prd_post = postproc(prd)
            prd = torch.from_numpy(prd_post).unsqueeze(0)
        else:
            prd = prd.squeeze(0)

        score = jaccard(mask, prd, reduce=None)
        scores.append(score)
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    return mean_scores, std_scores
