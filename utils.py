import os
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np


def transform_annotations(base_path: str, coco: COCO, mode: str) -> None:
    
    annot_path = os.path.join(base_path, mode + ".json")
    with open(annot_path, "r") as f:
        annots = json.load(f)

    new_annotations = []
    for ids, annot in enumerate(annots["annotations"]):
        img_id = annot["image_id"]
        img = coco.loadImgs(img_id)
        annot_ = annot.copy()
        
        seg = annot['segmentation'][0]
        if len(seg) < 4:
            continue
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        poly[:, 0] = poly[:, 0] * img[0]["width"]
        poly[:, 1] = poly[:, 1] * img[0]["height"]
        poly = poly.reshape(-1).tolist()
        annot_['segmentation'] = [poly]
        new_annotations.append(annot_)

    annots["annotations"] = new_annotations

    with open(os.path.join(base_path, mode + "_tf.json"), "w") as f:
        json.dump(annots, f, sort_keys=True, indent=4)