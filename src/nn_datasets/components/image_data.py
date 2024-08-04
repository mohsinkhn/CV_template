from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageData(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]
        row_id = row["id"]
        image = self._load_image(image_path)
        if self.transform:
            image = self.transform(image)
        return {"id": row_id, "image": image, "label": label}

    def _load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")
