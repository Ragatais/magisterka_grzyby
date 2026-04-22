import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_curve, auc



#MULTICLASS

def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #recall per class
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)

    #specificity per class
    cm = confusion_matrix(y_true, y_pred)
    specificity = []

    for i in range(num_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]

        if (tn + fp) == 0:
            specificity.append(0)
        else:
            specificity.append(tn / (tn + fp))

    specificity = np.array(specificity)

    balanced_acc = np.mean(recalls)

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # geometric mean
    gmean = np.sqrt(np.mean(recalls) * np.mean(specificity))

    return {
        "balanced_accuracy": balanced_acc,
        "f1_macro": f1,
        "recall_mean": np.mean(recalls),
        "specificity_mean": np.mean(specificity),
        "gmean": gmean,
        "confusion_matrix": cm
    }


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix", figsize=(10, 8)):
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        cmap="viridis",
        cbar=True,
        annot=False,
        xticklabels=False,
        yticklabels=False
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


#BINARY

def compute_metrics_binary(y_true, y_pred, y_score):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp
    p = tp + fn
    n = tn + fp

    accuracy = (tp + tn) / total if total > 0 else 0.0
    error = (fp + fn) / total if total > 0 else 0.0
    recall = tp / p if p > 0 else 0.0          # TPR, sensitivity
    specificity = tn / n if n > 0 else 0.0     # TNR
    fpr = fp / n if n > 0 else 0.0             # false positive rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0
    gmean = np.sqrt(recall * specificity)

    fpr_roc, tpr_roc, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr_roc, tpr_roc)

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "error": float(error),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "gmean": float(gmean),
        "confusion_matrix": cm,
        "roc_fpr": fpr_roc,
        "roc_tpr": tpr_roc,
        "roc_thresholds": thresholds,
        "roc_auc": float(roc_auc),
    }


def plot_confusion_matrix_binary(cm, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix - Binary Classifier")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guessing")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Binary Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


#OGÓLNE

def save_metrics_history(history, save_path):
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False, encoding="utf-8")