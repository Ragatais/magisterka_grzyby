import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    RAW_DATA_DIR,
    CHECKPOINTS_DIR,
    METADATA_DIR,
)

from model import build_resnet18
from transforms import get_eval_transforms


class HybridCascadeClassifier:
    def __init__(
        self,
        species_model_path=CHECKPOINTS_DIR / "best_model_species_known.pth",
        binary_model_path=CHECKPOINTS_DIR / "best_model_binary.pth",
        species_class_mapping_path=METADATA_DIR / "class_to_idx_species_known.csv",
        species_edibility_mapping_path=METADATA_DIR / "species_edibility.csv",
        species_conf_threshold=0.6,
        binary_conf_threshold=0.6,
        device=None,
    ):
        self.device = device or torch.device("cuda")

        self.species_conf_threshold = species_conf_threshold
        self.binary_conf_threshold = binary_conf_threshold

        self.transform = get_eval_transforms()

        self.species_class_mapping = self._load_species_class_mapping(
            species_class_mapping_path
        )
        self.idx_to_species = {
            idx: species
            for species, idx in self.species_class_mapping.items()
        }

        self.species_to_edibility = self._load_species_edibility_mapping(
            species_edibility_mapping_path
        )

        self.binary_label_map = {
            0: "edible",
            1: "poisonous"
        }

        self.species_model = self._load_species_model(species_model_path)
        self.binary_model = self._load_binary_model(binary_model_path)

    def _load_species_class_mapping(self, path):
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace("\ufeff", "", regex=False)

        df["species_label"] = df["species_label"].astype(str).str.strip()
        df["class_idx"] = df["class_idx"].astype(int)

        return dict(zip(df["species_label"], df["class_idx"]))

    def _load_species_edibility_mapping(self, path):
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace("\ufeff", "", regex=False)

        df["species_label"] = df["species_label"].astype(str).str.strip()
        df["edibility_label"] = (
            df["edibility_label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # tylko jednoznaczne klasy
        df = df[df["edibility_label"].isin(["edible", "poisonous"])]

        return dict(zip(df["species_label"], df["edibility_label"]))

    def _load_species_model(self, path):
        num_species_classes = len(self.species_class_mapping)

        model = build_resnet18(
            num_classes=num_species_classes,
            use_pretrained=False
        )

        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def _load_binary_model(self, path):
        model = build_resnet18(
            num_classes=2,
            use_pretrained=False
        )

        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def _load_image(self, relative_path):
        image_path = RAW_DATA_DIR / relative_path
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        return image

    def _predict_species(self, image_tensor):
        with torch.no_grad():
            logits = self.species_model(image_tensor)
            probs = F.softmax(logits, dim=1)

            confidence, pred_idx = torch.max(probs, dim=1)

        pred_idx = int(pred_idx.item())
        confidence = float(confidence.item())

        species_label = self.idx_to_species[pred_idx]
        edibility_from_species = self.species_to_edibility.get(species_label)

        return {
            "species_label": species_label,
            "species_class_idx": pred_idx,
            "species_confidence": confidence,
            "edibility_from_species": edibility_from_species,
        }

    def _predict_binary(self, image_tensor):
        with torch.no_grad():
            logits = self.binary_model(image_tensor)
            probs = F.softmax(logits, dim=1)

            confidence, pred_idx = torch.max(probs, dim=1)

        pred_idx = int(pred_idx.item())
        confidence = float(confidence.item())

        return {
            "binary_label": self.binary_label_map[pred_idx],
            "binary_class_idx": pred_idx,
            "binary_confidence": confidence,
        }

    def predict(self, relative_path):
        image_tensor = self._load_image(relative_path)

        species_result = self._predict_species(image_tensor)

        if (
            species_result["species_confidence"] >= self.species_conf_threshold
            and species_result["edibility_from_species"] is not None
        ):
            return {
                "final_decision": species_result["edibility_from_species"],
                "decision_source": "species_model",
                "rejected": False,
                **species_result,
                "binary_label": None,
                "binary_confidence": None,
            }

        binary_result = self._predict_binary(image_tensor)

        if binary_result["binary_confidence"] >= self.binary_conf_threshold:
            return {
                "final_decision": binary_result["binary_label"],
                "decision_source": "binary_model",
                "rejected": False,
                **species_result,
                **binary_result,
            }

        return {
            "final_decision": "reject",
            "decision_source": "reject",
            "rejected": True,
            **species_result,
            **binary_result,
        }