import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

from config import (
    SPLITS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    OUTPUTS_DIR,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    TUNED_NUM_EPOCHS,
    TUNED_LEARNING_RATE,
    TUNED_WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)

from utils import (
    compute_metrics_binary,
    plot_confusion_matrix_binary,
    plot_roc_curve,
    save_metrics_history
)


from dataset_binary import MushroomBinaryDataset
from transforms import get_train_transforms, get_eval_transforms
from model import build_resnet18


DEVICE = torch.device("cuda")

#trian loop

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())   # score klasy pozytywnej

    metrics = compute_metrics_binary(all_labels, all_preds, all_scores)

    return running_loss / len(loader), metrics


def main():
    train_csv = SPLITS_DIR / "train_with_edibility.csv"
    val_csv = SPLITS_DIR / "val_with_edibility.csv"

    train_dataset = MushroomBinaryDataset(
        train_csv,
        transform=get_train_transforms()
    )

    val_dataset = MushroomBinaryDataset(
        val_csv,
        transform=get_eval_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    num_classes = train_dataset.get_num_classes()

    model = build_resnet18(num_classes=num_classes, use_pretrained=False).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TUNED_LEARNING_RATE,
        weight_decay=TUNED_WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_val_score = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None
    history = []

    for epoch in range(TUNED_NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{TUNED_NUM_EPOCHS}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, metrics = evaluate(model, val_loader, criterion)

        scheduler.step(metrics["balanced_accuracy"])

        epoch_record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": metrics["accuracy"],
            "error": metrics["error"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "gmean": metrics["gmean"],
            "roc_auc": metrics["roc_auc"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        }
        history.append(epoch_record)

        save_metrics_history(
            history,
            METRICS_DIR / "metrics_binary.csv"
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Error: {metrics['error']:.4f}")
        print(f"Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall (TPR): {metrics['recall']:.4f}")
        print(f"Specificity (TNR): {metrics['specificity']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"G-mean: {metrics['gmean']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")

        # zapis najlepszego modelu
        if metrics["balanced_accuracy"] > best_val_score:
            best_val_score = metrics["balanced_accuracy"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(
                model.state_dict(),
                CHECKPOINTS_DIR / "best_model_binary.pth"
            )

            plot_confusion_matrix_binary(
                metrics["confusion_matrix"],
                FIGURES_DIR / "confusion_matrix_binary.png"
            )

            plot_roc_curve(
                metrics["roc_fpr"],
                metrics["roc_tpr"],
                metrics["roc_auc"],
                FIGURES_DIR / "roc_curve_binary.png"
            )

            best_metrics_df = pd.DataFrame([{
                "best_epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": metrics["accuracy"],
                "error": metrics["error"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "gmean": metrics["gmean"],
                "roc_auc": metrics["roc_auc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }])

            best_metrics_df.to_csv(
                METRICS_DIR / "best_metrics_binary.csv",
                index=False,
                encoding="utf-8"
            )

            print("Zapisano wyniki")
        else:
            epochs_without_improvement += 1
            print(f"Brak poprawy przez {epochs_without_improvement} epock.")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping activated at epoch {epoch + 1}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\nPodsumowanie")
    print(f"Best epoch: {best_epoch}")
    print(f"Best balanced accuracy: {best_val_score:.4f}")


if __name__ == "__main__":
    main()