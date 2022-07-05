import os
import torch
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import io, transforms
import torchvision.transforms.functional as tf
from typing import Any, Callable, List, Optional, Tuple


def resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    mask = F.interpolate(mask.unsqueeze(0), size, mode="nearest").squeeze(0)
    return mask


class COCODataset(Dataset):
    def __init__(
        self, coco_annot: COCO, root_path: str, transform: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.cat_ids = coco_annot.getCatIds()
        self.root_path = root_path
        self.coco_annot = coco_annot
        self.img_ids = self.coco_annot.getImgIds(catIds=self.coco_annot.getCatIds())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        img_id = self.img_ids[i]

        annIds = self.coco_annot.getAnnIds(
            imgIds=img_id, catIds=self.cat_ids, iscrowd=None
        )
        anns = self.coco_annot.loadAnns(annIds)

        img = self.coco_annot.loadImgs(img_id)[0]
        img = io.read_image(os.path.join(self.root_path, "data", img["file_name"]))
        if img.shape[0] == 1:
            img = torch.cat([img] * 3)

        mask = torch.tensor(
            np.max(
                np.stack(
                    [
                        self.coco_annot.annToMask(ann) * ann["category_id"]
                        for ann in anns
                    ]
                ),
                axis=0,
            )
        )

        if self.transform is not None:
            return self.transform(img, mask)

        mask = torch.tensor(mask).type(torch.long)
        return img, mask


class Transformations:
    def __init__(
        self,
        img_size: Tuple[int, int],
        p_hflip: float,
        p_vflip: float,
        p_invert: float,
        p_rgb_to_gs: float,
        mask_size: Tuple[int, int],
    ):
        self.resize = transforms.Resize(img_size)
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_invert = p_invert
        self.p_rgb_to_gs = p_rgb_to_gs
        self.mask_size = mask_size

    def __call__(self, img, mask):

        img = self.resize(img)
        mask = self.resize(mask.unsqueeze(0))

        if np.random.rand() < self.p_hflip:
            img = tf.hflip(img)
            mask = tf.hflip(mask)

        if np.random.rand() < self.p_vflip:
            img = tf.vflip(img)
            mask = tf.vflip(mask)

        if np.random.rand() < self.p_invert:
            img = tf.invert(img)

        if np.random.rand() < self.p_rgb_to_gs:
            img = tf.rgb_to_grayscale(img, num_output_channels=3)

        resized_mask = resize_mask(mask, self.mask_size)

        img = img.type(torch.float32) / 255
        mask = mask.type(torch.long)
        resized_mask = resized_mask.type(torch.long)

        return img, mask, resized_mask


class COCODatasetInference(Dataset):
    def __init__(
        self, coco_annot: COCO, root_path: str, img_size: Tuple[int, int]
    ) -> None:
        super().__init__()
        self.cat_ids = coco_annot.getCatIds()
        self.root_path = root_path
        self.coco_annot = coco_annot
        self.img_ids = self.coco_annot.getImgIds(catIds=self.coco_annot.getCatIds())
        self.resize = transforms.Resize(img_size)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        img_id = self.img_ids[i]

        img = self.coco_annot.loadImgs(img_id)[0]
        orig_img = io.read_image(os.path.join(self.root_path, "data", img["file_name"]))

        img = self.resize(orig_img)

        img = img.type(torch.float32) / 255

        return img, orig_img
