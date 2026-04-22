import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data._utils import pin_memory
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
    METRICS_DIR
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
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
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
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

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
        num_workers=NUM_WORKERS
    )

    num_classes = train_dataset.get_num_classes()

    model = build_resnet18(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_val_score = 0.0

    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, metrics = evaluate(model, val_loader, criterion, num_classes)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "recall_mean": metrics["recall_mean"],
            "specificity_mean": metrics["specificity_mean"],
            "gmean": metrics["gmean"],
        }
        history.append(epoch_record)

        save_metrics_history(
            history,
            METRICS_DIR / "metrics_baseline.csv"
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

            torch.save(
                model.state_dict(),
                CHECKPOINTS_DIR / "best_model_baseline.pth"
            )

            plot_confusion_matrix(
                metrics["confusion_matrix"],
                FIGURES_DIR / "confusion_matrix_baseline.png",
                title="Confusion Matrix - Baseline Model"
            )

            best_metrics_df = pd.DataFrame([{
                "best_epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "recall_mean": metrics["recall_mean"],
                "specificity_mean": metrics["specificity_mean"],
                "f1_macro": metrics["f1_macro"],
                "gmean": metrics["gmean"],
            }])

            best_metrics_df.to_csv(
                METRICS_DIR / "best_metrics_baseline.csv",
                index=False,
                encoding="utf-8"
            )

            print("Zapisano wyniki")


if __name__ == "__main__":
    main()