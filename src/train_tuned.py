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
    f1_score,
    recall_score
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
    compute_metrics,
    plot_confusion_matrix,
    save_metrics_history
)


from dataset import MushroomDataset
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


def evaluate(model, loader, criterion, num_classes):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, num_classes)

    return running_loss / len(loader), metrics


def main():
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    train_dataset = MushroomDataset(
        train_csv,
        transform=get_train_transforms()
    )

    class_to_idx = train_dataset.class_to_idx

    val_dataset = MushroomDataset(
        val_csv,
        transform=get_eval_transforms(),
        class_to_idx=class_to_idx
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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

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
        val_loss, metrics = evaluate(model, val_loader, criterion, num_classes)

        scheduler.step(metrics["balanced_accuracy"])

        epoch_record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "recall_mean": metrics["recall_mean"],
            "specificity_mean": metrics["specificity_mean"],
            "f1_macro": metrics["f1_macro"],
            "gmean": metrics["gmean"],
        }

        history.append(epoch_record)

        save_metrics_history(
            history,
            METRICS_DIR / "metrics_tuned.csv"
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        print(f"Recall mean: {metrics['recall_mean']:.4f}")
        print(f"Specificity mean: {metrics['specificity_mean']:.4f}")
        print(f"F1 macro: {metrics['f1_macro']:.4f}")
        print(f"G-mean: {metrics['gmean']:.4f}")

        # zapis najlepszego modelu
        if metrics["balanced_accuracy"] > best_val_score:
            best_val_score = metrics["balanced_accuracy"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(
                model.state_dict(),
                CHECKPOINTS_DIR / "best_model_tuned.pth"
            )

            plot_confusion_matrix(
                metrics["confusion_matrix"],
                FIGURES_DIR / "confusion_matrix_tuned.png",
                title="Confusion Matrix - Tuned Model"
            )

            best_metrics_df = pd.DataFrame([{
                "best_epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "recall_mean": metrics["recall_mean"],
                "specificity_mean": metrics["specificity_mean"],
                "f1_macro": metrics["f1_macro"],
                "gmean": metrics["gmean"],
            }])

            best_metrics_df.to_csv(
                METRICS_DIR / "best_metrics_tuned.csv",
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