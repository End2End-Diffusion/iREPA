import os

from datasets import load_from_disk
import numpy as np
import torch
from torch.utils.data import Dataset


class HFImageDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        split_str = "val" if split == "val" else ""
        self.img_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-images", split_str))
        self.transform = transform

    def __getitem__(self, idx):
        img_elem = self.img_dataset[idx]
        image, label = img_elem["image"], img_elem["label"]

        image = np.array(image.convert("RGB")).transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]

        image = torch.from_numpy(image)
        label = torch.tensor(label)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_dataset)
