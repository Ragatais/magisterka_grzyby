import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from config import (
    SPLITS_DIR,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_CLASSES,
)

from dataset import MushroomDataset
from transforms import get_eval_transforms
from model import build_resnet18
from utils import compute_metrics, plot_confusion_matrix


DEVICE = torch.device("cuda")


def main():
    print("Device:", DEVICE)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test_csv = SPLITS_DIR / "test.csv"

    test_dataset = MushroomDataset(
        csv_path=test_csv,
        transform=get_eval_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = build_resnet18(num_classes=NUM_CLASSES, use_pretrained=False)
    model.load_state_dict(
        torch.load(CHECKPOINTS_DIR / "best_model_tuned.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test multiclass tuned"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred, NUM_CLASSES)

    metrics_df = pd.DataFrame([{
        "balanced_accuracy": metrics["balanced_accuracy"],
        "recall_mean": metrics["recall_mean"],
        "specificity_mean": metrics["specificity_mean"],
        "f1_macro": metrics["f1_macro"],
        "gmean": metrics["gmean"],
    }])

    metrics_df.to_csv(
        METRICS_DIR / "test_metrics_multiclass_tuned.csv",
        index=False,
        encoding="utf-8"
    )

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        FIGURES_DIR / "confusion_matrix_multiclass_tuned_test.png",
        title="Confusion Matrix - Multiclass Tuned Model",
        figsize=(12, 10)
    )

    print("\nPodsumowanie modelu wieloklasowego")
    print(metrics_df)


if __name__ == "__main__":
    main()