import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config import RAW_DATA_DIR


class MushroomBinaryDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)

        # usunięcie unknown
        self.df = self.df[self.df["edibility_label"] != "unknown"].reset_index(drop=True)

        self.transform = transform

        # mapowanie binarne
        self.label_map = {
            "edible": 0,
            "poisonous": 1
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = RAW_DATA_DIR / row["relative_path"]
        image = Image.open(image_path).convert("RGB")

        label = self.label_map[row["edibility_label"]]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_num_classes(self):
        return 2