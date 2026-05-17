'''
Informacja o wykorzystaniu AI:
Ten plik został opracowany przy wsparciu asystenta
ChatGPT (OpenAI, modele GPT-5 oraz GPT-5.5).
'''

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config import RAW_DATA_DIR


class MushroomDataset(Dataset):
    def __init__(self, csv_path, transform=None, class_to_idx=None):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.transform = transform

        self.classes = sorted(self.data["species_label"].unique())

        if class_to_idx is None:
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx

        self.data["class_idx"] = self.data["species_label"].map(self.class_to_idx)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = RAW_DATA_DIR / row["relative_path"]
        label = int(row["class_idx"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_num_classes(self):
        return len(self.class_to_idx)