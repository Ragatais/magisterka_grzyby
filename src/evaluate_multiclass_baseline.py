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
        torch.load(CHECKPOINTS_DIR / "best_model_baseline.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    prediction_records = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test multiclass baseline"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            batch_paths = test_dataset.data.iloc[
                len(y_true): len(y_true) + len(labels)
            ]["relative_path"].tolist()

            batch_true_labels = test_dataset.data.iloc[
                len(y_true): len(y_true) + len(labels)
            ]["species_label"].tolist()

            idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

            for path, true_label_name, true_idx, pred_idx in zip(
                    batch_paths,
                    batch_true_labels,
                    labels.cpu().numpy(),
                    preds.cpu().numpy()
            ):
                pred_label_name = idx_to_class[int(pred_idx)]

                prediction_records.append({
                    "relative_path": path,
                    "true_label": true_label_name,
                    "pred_label": pred_label_name,
                    "true_idx": int(true_idx),
                    "pred_idx": int(pred_idx),
                    "correct": int(true_idx) == int(pred_idx),
                })

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
        METRICS_DIR / "test_metrics_multiclass_baseline.csv",
        index=False,
        encoding="utf-8"
    )

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        FIGURES_DIR / "confusion_matrix_multiclass_baseline_test.png",
        title="Confusion Matrix - Multiclass Baseline Model",
        figsize=(12, 10)
    )

    pd.DataFrame(prediction_records).to_csv(
        METRICS_DIR / "test_predictions_multiclass_baseline.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nPodsumowanie modelu wieloklasowego")
    print(metrics_df)


if __name__ == "__main__":
    main()