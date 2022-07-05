import os
import cv2
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pycocotools.coco import COCO


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform_annotations(base_path: str, coco: COCO, mode: str) -> None:

    annot_path = os.path.join(base_path, mode + ".json")
    with open(annot_path, "r") as f:
        annots = json.load(f)

    new_annotations = []
    for ids, annot in enumerate(annots["annotations"]):
        img_id = annot["image_id"]
        img = coco.loadImgs(img_id)
        annot_ = annot.copy()

        seg = annot["segmentation"][0]
        if len(seg) < 4:
            continue
        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        poly[:, 0] = poly[:, 0] * img[0]["width"]
        poly[:, 1] = poly[:, 1] * img[0]["height"]
        poly = poly.reshape(-1).tolist()
        annot_["segmentation"] = [poly]
        new_annotations.append(annot_)

    annots["annotations"] = new_annotations

    with open(os.path.join(base_path, mode + "_tf.json"), "w") as f:
        json.dump(annots, f, sort_keys=True, indent=4)


def load_model(weights, model, device):
    model = model.to(device)
    checkpoint = torch.load(weights, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def plot_batch_sample_inference(model, batch, device):

    img, mask, _ = batch

    prd = infer_img(model, img, device)

    img = img.moveaxis(1, 3)
    mask = mask.squeeze(1)

    bs = img.shape[0]
    fig, ax = plt.subplots(bs, 3, figsize=(8, 8), sharey=True)

    if bs == 1:
        ax[0].imshow(img[0, ...].detach().cpu().numpy())
        ax[1].imshow(mask[0, ...].detach().cpu().numpy())
        ax[2].imshow(prd[0, ...])
        ax[0].set_title("Input image")
        ax[1].set_title("GT mask")
        ax[2].set_title("Prd mask")

    else:
        for i in range(bs):
            ax[i][0].imshow(img[i, ...].detach().cpu().numpy())
            ax[i][1].imshow(mask[i, ...].detach().cpu().numpy())
            ax[i][2].imshow(prd[i, ...])

        ax[0][0].set_title("Input image")
        ax[0][1].set_title("GT mask")
        ax[0][2].set_title("Prd mask")

    plt.tight_layout()


def infer_img(model, img, device):
    prd = model(img.to(device))
    prd = torch.argmax(prd, dim=1)
    prd_res = (
        F.interpolate(
            prd.unsqueeze(0).type(torch.float32),
            (img.shape[2], img.shape[3]),
            mode="nearest",
        )
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    return prd_res


class ErodeDilate:
    def __init__(self, kernel_size, erode_iter, dilate_iter):
        self.erode_iter = erode_iter
        self.dilate_iter = dilate_iter
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, mask):
        mask = mask.detach().cpu().numpy().squeeze(0).squeeze(0)
        mask = cv2.erode(mask, self.kernel, iterations=self.erode_iter)
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilate_iter)
        return mask