import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

from config import (
    SPLITS_DIR,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from dataset_binary import MushroomBinaryDataset
from transforms import get_eval_transforms
from model import build_resnet18
from utils import (
    compute_metrics_binary,
    plot_confusion_matrix_binary,
    plot_roc_curve,
)


DEVICE = torch.device("cuda")


def main():
    print("Device:", DEVICE)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test_csv = SPLITS_DIR / "test_known_edibility.csv"

    test_dataset = MushroomBinaryDataset(
        csv_file=test_csv,
        transform=get_eval_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = build_resnet18(num_classes=2, use_pretrained=False)
    model.load_state_dict(
        torch.load(CHECKPOINTS_DIR / "best_model_binary.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    prediction_records = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test binary"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)
            positive_scores = probs[:, 1]

            batch_paths = test_dataset.df.iloc[
                len(y_true): len(y_true) + len(labels)
            ]["relative_path"].tolist()

            idx_to_label = {
                0: "edible",
                1: "poisonous"
            }

            for path, true_idx, pred_idx, score in zip(
                    batch_paths,
                    labels.cpu().numpy(),
                    preds.cpu().numpy(),
                    positive_scores.cpu().numpy()
            ):
                prediction_records.append({
                    "relative_path": path,
                    "true_label": idx_to_label[int(true_idx)],
                    "pred_label": idx_to_label[int(pred_idx)],
                    "true_idx": int(true_idx),
                    "pred_idx": int(pred_idx),
                    "score_poisonous": float(score),
                    "correct": int(true_idx) == int(pred_idx),
                })

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(positive_scores.cpu().numpy())

    metrics = compute_metrics_binary(y_true, y_pred, y_score)

    metrics_df = pd.DataFrame([{
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "accuracy": metrics["accuracy"],
        "error": metrics["error"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "fpr": metrics["fpr"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "gmean": metrics["gmean"],
        "roc_auc": metrics["roc_auc"],
    }])

    metrics_df.to_csv(
        METRICS_DIR / "test_metrics_binary.csv",
        index=False,
        encoding="utf-8"
    )

    plot_confusion_matrix_binary(
        metrics["confusion_matrix"],
        FIGURES_DIR / "confusion_matrix_binary_test.png"
    )

    plot_roc_curve(
        metrics["roc_fpr"],
        metrics["roc_tpr"],
        metrics["roc_auc"],
        FIGURES_DIR / "roc_curve_binary_test.png"
    )

    pd.DataFrame(prediction_records).to_csv(
        METRICS_DIR / "test_predictions_binary.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nPodsumowanie modelu binarnego")
    print(metrics_df)


if __name__ == "__main__":
    main()