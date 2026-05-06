import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

from config import (
    SPLITS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
)
from hybrid_classifier import HybridCascadeClassifier

import matplotlib.pyplot as plt
import seaborn as sns


LABEL_TO_IDX = {
    "edible": 0,
    "poisonous": 1,
}


SPECIES_THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80]
BINARY_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]


def plot_confusion_matrix(cm, save_path, title):
    plt.figure(figsize=(12, 9))

    sns.heatmap(
        cm,
        cmap="Blues",
        cbar=True,
        annot=True,
        fmt="d",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        annot_kws={"size": 18}
    )

    plt.title(title, fontsize=22, pad=18)
    plt.xlabel("Predicted class", fontsize=18)
    plt.ylabel("True class", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_balanced_accuracy_coverage_tradeoff(results_df, save_path):
    plt.figure(figsize=(10, 7))

    plt.scatter(
        results_df["coverage"],
        results_df["balanced_accuracy"],
        s=80
    )

    for _, row in results_df.iterrows():
        label = f"{row['species_threshold']:.1f}/{row['binary_threshold']:.1f}"
        plt.annotate(
            label,
            (row["coverage"], row["balanced_accuracy"]),
            fontsize=9,
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.title("Hybrid Cascade: Balanced Accuracy vs Coverage", fontsize=18, pad=15)
    plt.xlabel("Coverage", fontsize=14)
    plt.ylabel("Balanced accuracy", fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_metrics(y_true, y_pred, rejected_flags):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rejected_flags = np.array(rejected_flags)

    rejected_count = int(rejected_flags.sum())
    accepted_mask = ~rejected_flags

    coverage = float(accepted_mask.mean())
    rejection_rate = float(rejected_flags.mean())

    if accepted_mask.sum() == 0:
        return {
            "coverage": coverage,
            "rejection_rate": rejection_rate,
            "accepted_count": 0,
            "rejected_count": rejected_count,
            "balanced_accuracy": 0.0,
            "recall_mean": 0.0,
            "specificity_mean": 0.0,
            "f1_macro": 0.0,
            "gmean": 0.0,
            "confusion_matrix": np.zeros((2, 2), dtype=int),
        }

    y_true_acc = y_true[accepted_mask]
    y_pred_acc = y_pred[accepted_mask]

    recalls = recall_score(
        y_true_acc,
        y_pred_acc,
        average=None,
        labels=[0, 1],
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true_acc,
        y_pred_acc,
        labels=[0, 1],
    )

    specificity = []
    total_cm = np.sum(cm)

    for i in range(2):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_cm - tp - fn - fp

        if tn + fp == 0:
            specificity.append(0.0)
        else:
            specificity.append(tn / (tn + fp))

    specificity = np.array(specificity)

    recall_mean = float(np.mean(recalls))
    specificity_mean = float(np.mean(specificity))

    return {
        "coverage": coverage,
        "rejection_rate": rejection_rate,
        "accepted_count": int(accepted_mask.sum()),
        "rejected_count": rejected_count,
        "balanced_accuracy": float(balanced_accuracy_score(y_true_acc, y_pred_acc)),
        "recall_mean": recall_mean,
        "specificity_mean": specificity_mean,
        "f1_macro": float(f1_score(y_true_acc, y_pred_acc, average="macro", zero_division=0)),
        "gmean": float(np.sqrt(recall_mean * specificity_mean)),
        "confusion_matrix": cm,
    }


def evaluate_split(csv_path, species_threshold, binary_threshold, save_predictions_path=None):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    classifier = HybridCascadeClassifier(
        species_conf_threshold=species_threshold,
        binary_conf_threshold=binary_threshold,
    )

    y_true = []
    y_pred = []
    rejected_flags = []
    prediction_records = []

    for _, row in df.iterrows():
        relative_path = row["relative_path"]
        true_label = row["edibility_label"]

        result = classifier.predict(relative_path)

        y_true.append(LABEL_TO_IDX[true_label])

        if result["rejected"]:
            y_pred.append(-1)
            rejected_flags.append(True)
        else:
            y_pred.append(LABEL_TO_IDX[result["final_decision"]])
            rejected_flags.append(False)

        prediction_records.append({
            "relative_path": relative_path,
            "true_label": true_label,
            "final_decision": result["final_decision"],
            "decision_source": result["decision_source"],
            "rejected": result["rejected"],
            "species_label": result["species_label"],
            "species_confidence": result["species_confidence"],
            "edibility_from_species": result["edibility_from_species"],
            "binary_label": result["binary_label"],
            "binary_confidence": result["binary_confidence"],
        })

    metrics = compute_metrics(y_true, y_pred, rejected_flags)

    if save_predictions_path is not None:
        pd.DataFrame(prediction_records).to_csv(
            save_predictions_path,
            index=False,
            encoding="utf-8",
        )

    return metrics


def build_selected_configs(results_df):
    selected_configs = []

    best_quality = results_df.sort_values(
        by=["balanced_accuracy", "coverage", "f1_macro"],
        ascending=[False, False, False],
    ).iloc[0]

    selected_configs.append({
        "variant": "best_quality",
        "species_threshold": float(best_quality["species_threshold"]),
        "binary_threshold": float(best_quality["binary_threshold"]),
        "selection_metric": "max_balanced_accuracy",
        "validation_balanced_accuracy": float(best_quality["balanced_accuracy"]),
        "validation_f1_macro": float(best_quality["f1_macro"]),
        "validation_gmean": float(best_quality["gmean"]),
        "validation_coverage": float(best_quality["coverage"]),
        "validation_rejection_rate": float(best_quality["rejection_rate"]),
        "validation_balanced_accuracy_coverage_score": float(
            best_quality["balanced_accuracy_coverage_score"]
        ),
    })

    best_compromise = results_df.sort_values(
        by=["balanced_accuracy_coverage_score", "balanced_accuracy", "coverage"],
        ascending=[False, False, False],
    ).iloc[0]

    selected_configs.append({
        "variant": "best_compromise",
        "species_threshold": float(best_compromise["species_threshold"]),
        "binary_threshold": float(best_compromise["binary_threshold"]),
        "selection_metric": "max_balanced_accuracy_times_coverage",
        "validation_balanced_accuracy": float(best_compromise["balanced_accuracy"]),
        "validation_f1_macro": float(best_compromise["f1_macro"]),
        "validation_gmean": float(best_compromise["gmean"]),
        "validation_coverage": float(best_compromise["coverage"]),
        "validation_rejection_rate": float(best_compromise["rejection_rate"]),
        "validation_balanced_accuracy_coverage_score": float(
            best_compromise["balanced_accuracy_coverage_score"]
        ),
    })

    coverage_min = 0.70
    candidates = results_df[results_df["coverage"] >= coverage_min]

    if not candidates.empty:
        best_constrained = candidates.sort_values(
            by=["balanced_accuracy", "coverage", "f1_macro"],
            ascending=[False, False, False],
        ).iloc[0]

        selected_configs.append({
            "variant": "best_constrained_coverage_0_70",
            "species_threshold": float(best_constrained["species_threshold"]),
            "binary_threshold": float(best_constrained["binary_threshold"]),
            "selection_metric": "max_balanced_accuracy_with_coverage_at_least_0_70",
            "validation_balanced_accuracy": float(best_constrained["balanced_accuracy"]),
            "validation_f1_macro": float(best_constrained["f1_macro"]),
            "validation_gmean": float(best_constrained["gmean"]),
            "validation_coverage": float(best_constrained["coverage"]),
            "validation_rejection_rate": float(best_constrained["rejection_rate"]),
            "validation_balanced_accuracy_coverage_score": float(
                best_constrained["balanced_accuracy_coverage_score"]
            ),
        })
    else:
        print("Brak konfiguracji z coverage >= 0.70")

    return selected_configs


def validation_grid_search():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    val_csv = SPLITS_DIR / "val_known_edibility.csv"

    results = []

    for species_threshold in SPECIES_THRESHOLDS:
        for binary_threshold in BINARY_THRESHOLDS:
            print(
                f"\nVAL | T_species={species_threshold:.2f}, "
                f"T_binary={binary_threshold:.2f}"
            )

            metrics = evaluate_split(
                val_csv,
                species_threshold=species_threshold,
                binary_threshold=binary_threshold,
            )

            row = {
                "species_threshold": species_threshold,
                "binary_threshold": binary_threshold,
                "coverage": metrics["coverage"],
                "rejection_rate": metrics["rejection_rate"],
                "accepted_count": metrics["accepted_count"],
                "rejected_count": metrics["rejected_count"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "recall_mean": metrics["recall_mean"],
                "specificity_mean": metrics["specificity_mean"],
                "f1_macro": metrics["f1_macro"],
                "gmean": metrics["gmean"],
            }

            results.append(row)

            print(
                f"Balanced Acc: {metrics['balanced_accuracy']:.4f} | "
                f"F1 macro: {metrics['f1_macro']:.4f} | "
                f"Coverage: {metrics['coverage']:.4f} | "
                f"Reject: {metrics['rejection_rate']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df["balanced_accuracy_coverage_score"] = (
        results_df["balanced_accuracy"] * results_df["coverage"]
    )

    results_path = METRICS_DIR / "hybrid_val_grid_search.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")

    plot_balanced_accuracy_coverage_tradeoff(
        results_df,
        FIGURES_DIR / "hybrid_balanced_accuracy_coverage_tradeoff.png"
    )

    selected_configs = build_selected_configs(results_df)

    selected_configs_df = pd.DataFrame(selected_configs)

    selected_configs_df.to_csv(
        METRICS_DIR / "hybrid_selected_configs.csv",
        index=False,
        encoding="utf-8"
    )

    config_path = METRICS_DIR / "hybrid_selected_configs.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(selected_configs, f, indent=4, ensure_ascii=False)

    print("\nNajlepsze konfiguracje:")
    print(selected_configs_df)

    return selected_configs


def test_selected_configs(selected_configs):
    test_csv = SPLITS_DIR / "test_known_edibility.csv"

    all_test_results = []

    for config in selected_configs:
        variant = config["variant"]
        species_threshold = config["species_threshold"]
        binary_threshold = config["binary_threshold"]

        print(
            f"\nTEST | variant={variant} | "
            f"T_species={species_threshold:.2f}, "
            f"T_binary={binary_threshold:.2f}"
        )

        predictions_path = METRICS_DIR / f"hybrid_test_predictions_{variant}.csv"

        metrics = evaluate_split(
            test_csv,
            species_threshold=species_threshold,
            binary_threshold=binary_threshold,
            save_predictions_path=predictions_path,
        )

        test_result = {
            "variant": variant,
            "selection_metric": config["selection_metric"],
            "species_threshold": species_threshold,
            "binary_threshold": binary_threshold,
            "validation_balanced_accuracy": config["validation_balanced_accuracy"],
            "validation_f1_macro": config["validation_f1_macro"],
            "validation_gmean": config["validation_gmean"],
            "validation_coverage": config["validation_coverage"],
            "validation_rejection_rate": config["validation_rejection_rate"],
            "validation_balanced_accuracy_coverage_score": config[
                "validation_balanced_accuracy_coverage_score"
            ],
            "test_coverage": metrics["coverage"],
            "test_rejection_rate": metrics["rejection_rate"],
            "test_accepted_count": metrics["accepted_count"],
            "test_rejected_count": metrics["rejected_count"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_recall_mean": metrics["recall_mean"],
            "test_specificity_mean": metrics["specificity_mean"],
            "test_f1_macro": metrics["f1_macro"],
            "test_gmean": metrics["gmean"],
            "test_balanced_accuracy_coverage_score": (
                metrics["balanced_accuracy"] * metrics["coverage"]
            ),
        }

        all_test_results.append(test_result)

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            FIGURES_DIR / f"confusion_matrix_hybrid_test_{variant}.png",
            title=f"Confusion Matrix - Hybrid ({variant})",
        )

        print("\nWynik testowy:")
        print(test_result)

    test_results_df = pd.DataFrame(all_test_results)

    test_results_df.to_csv(
        METRICS_DIR / "hybrid_test_results_all_variants.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nNajlepsze wyniki (hybryda/test):")
    print(test_results_df)

    return test_results_df


def main():
    selected_configs = validation_grid_search()
    test_selected_configs(selected_configs)


if __name__ == "__main__":
    main()