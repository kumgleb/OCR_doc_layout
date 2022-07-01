import os
import torch
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import torchvision.transforms.functional as tf
from tqdm import tqdm
from typing import Any, Callable, List, Optional, Tuple


class COCODataset(Dataset):
    def __init__(
        self, 
        coco_annot: COCO, 
        root_path: str, 
        transform: Optional[Callable]=None
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

        annIds = self.coco_annot.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
        anns = self.coco_annot.loadAnns(annIds)
        
        img = self.coco_annot.loadImgs(img_id)[0]
        img = io.read_image(os.path.join(self.root_path, "data", img['file_name']))
        if img.shape[0] == 1:
            img = torch.cat([img]*3)

        mask = np.zeros((img.shape[1], img.shape[2]))
        for i in range(len(anns)):
            pixel_value = int(anns[i]['category_id'])
            
            mask = np.maximum(self.coco_annot.annToMask(anns[i])*pixel_value, mask)
        
        if self.transform is not None:
            return self.transform(img, mask)
        
        return img, mask