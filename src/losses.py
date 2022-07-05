from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F


def dice(prd_logits, y, eps=1):
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
