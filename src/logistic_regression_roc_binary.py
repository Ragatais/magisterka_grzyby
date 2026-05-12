import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from config import METRICS_DIR, STATISTICAL_TESTS_DIR


POSITIVE_CLASS = "poisonous"
NEGATIVE_CLASS = "edible"

SCORE_COL = "score_poisonous"
TRUE_LABEL_COL = "true_label"


def load_binary_predictions():
    input_path = METRICS_DIR / "test_predictions_binary.csv"
    df = pd.read_csv(input_path)

    required_columns = [TRUE_LABEL_COL, SCORE_COL]

    df = df.dropna(subset=required_columns).copy()

    df["y_true"] = (df[TRUE_LABEL_COL] == POSITIVE_CLASS).astype(int)
    df["y_score"] = df[SCORE_COL].astype(float)

    return df


def calculate_metrics_for_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr = 1 - specificity if not np.isnan(specificity) else np.nan

    gmean = np.sqrt(sensitivity * specificity)

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "gmean": gmean,
        "youden_j": sensitivity - fpr,
    }


def build_threshold_grid(y_true, y_score):
    thresholds = np.linspace(0.0, 1.0, 1001)

    results = [
        calculate_metrics_for_threshold(y_true, y_score, threshold)
        for threshold in thresholds
    ]

    return pd.DataFrame(results)


def select_optimal_thresholds(threshold_metrics):
    rows = []

    default_row = threshold_metrics.iloc[
        (threshold_metrics["threshold"] - 0.5).abs().argsort().iloc[0]
    ].to_dict()

    default_row.update({
        "criterion": "default_threshold_0_5",
        "description": "Domyślny próg odcięcia równy 0.5",
    })
    rows.append(default_row)

    youden_row = threshold_metrics.loc[
        threshold_metrics["youden_j"].idxmax()
    ].to_dict()

    youden_row.update({
        "criterion": "max_youden_j",
        "description": "Maksymalizacja indeksu Youdena: sensitivity - false positive rate",
    })
    rows.append(youden_row)

    gmean_row = threshold_metrics.loc[
        threshold_metrics["gmean"].idxmax()
    ].to_dict()

    gmean_row.update({
        "criterion": "max_gmean",
        "description": "Maksymalizacja średniej geometrycznej sensitivity i specificity",
    })
    rows.append(gmean_row)

    screening_candidates = threshold_metrics[
        threshold_metrics["recall_sensitivity"] >= 0.90
    ]

    if len(screening_candidates) > 0:
        screening_row = screening_candidates.loc[
            screening_candidates["specificity"].idxmax()
        ].to_dict()

        screening_row.update({
            "criterion": "screening_sensitivity_at_least_0_90",
            "description": "Próg dla wariantu przesiewowego: sensitivity >= 0.90 oraz maksymalna specificity",
        })
        rows.append(screening_row)

    confirmation_candidates = threshold_metrics[
        threshold_metrics["specificity"] >= 0.95
    ]

    if len(confirmation_candidates) > 0:
        confirmation_row = confirmation_candidates.loc[
            confirmation_candidates["recall_sensitivity"].idxmax()
        ].to_dict()

        confirmation_row.update({
            "criterion": "confirmation_specificity_at_least_0_95",
            "description": "Próg dla wariantu potwierdzającego: specificity >= 0.95 oraz maksymalna sensitivity",
        })
        rows.append(confirmation_row)

    return pd.DataFrame(rows)


def fit_logistic_regression(df):
    x = df[["y_score"]].values
    y = df["y_true"].values

    model = LogisticRegression()
    model.fit(x, y)

    df = df.copy()
    df["logistic_regression_probability"] = model.predict_proba(x)[:, 1]

    coefficients = pd.DataFrame([{
        "intercept": model.intercept_[0],
        "coefficient_score_poisonous": model.coef_[0][0],
    }])

    return df, coefficients, model


def plot_logistic_regression_curve(df, model, figures_dir):
    x_grid = np.linspace(0, 1, 500).reshape(-1, 1)
    y_probability = model.predict_proba(x_grid)[:, 1]

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="y_score",
        y="y_true",
        alpha=0.25,
        label="Obserwacje testowe"
    )

    plt.plot(
        x_grid.ravel(),
        y_probability,
        linewidth=2,
        label="Regresja logistyczna"
    )

    plt.axhline(
        y=0.5,
        linestyle="--",
        linewidth=1.5,
        label="Próg decyzyjny p = 0.5"
    )

    plt.title("Regresja logistyczna dla klasy poisonous")
    plt.xlabel("Score poisonous")
    plt.ylabel("Prawdopodobieństwo klasy poisonous")

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )

    plt.tight_layout()

    output_path = figures_dir / "logistic_regression_score_poisonous.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_logistic_threshold_effect(df, figures_dir):
    y_true = df["y_true"].values
    y_score = df["logistic_regression_probability"].values

    threshold_metrics = build_threshold_grid(y_true, y_score)

    plt.figure(figsize=(10, 6))

    plt.plot(
        threshold_metrics["threshold"],
        threshold_metrics["balanced_accuracy"],
        label="Balanced accuracy"
    )

    plt.plot(
        threshold_metrics["threshold"],
        threshold_metrics["recall_sensitivity"],
        label="Sensitivity"
    )

    plt.plot(
        threshold_metrics["threshold"],
        threshold_metrics["specificity"],
        label="Specificity"
    )

    plt.plot(
        threshold_metrics["threshold"],
        threshold_metrics["gmean"],
        label="G-mean"
    )

    plt.axvline(
        x=0.5,
        linestyle="--",
        linewidth=1.5,
        label="Domyślny próg p = 0.5"
    )

    plt.title("Wpływ progu prawdopodobieństwa regresji logistycznej na metryki")
    plt.xlabel("Próg prawdopodobieństwa")
    plt.ylabel("Wartość metryki")

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )

    plt.tight_layout()

    output_path = figures_dir / "logistic_regression_metrics_vs_probability_threshold.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return threshold_metrics

def plot_roc_curve_with_thresholds(y_true, y_score, optimal_thresholds, figures_dir):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 7))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC, AUC = {auc_value:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Klasyfikator losowy"
    )

    for _, row in optimal_thresholds.iterrows():
        plt.scatter(
            row["fpr"],
            row["recall_sensitivity"],
            s=70,
            label=f"{row['criterion']} (t={row['threshold']:.3f})"
        )

    plt.title("Krzywa ROC z wybranymi progami odcięcia")
    plt.xlabel("False Positive Rate, czyli 1 - specificity")
    plt.ylabel("True Positive Rate, czyli sensitivity")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    output_path = figures_dir / "roc_curve_with_optimal_thresholds.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metrics_vs_threshold(threshold_metrics, optimal_thresholds, figures_dir):
    plt.figure(figsize=(10, 6))

    plt.plot(
        threshold_metrics["threshold"],
        threshold_metrics["balanced_accuracy"],
        label="Balanced accuracy"
    )

    plt.plot(
        threshold_metrics["threshold"],
        threshold_metrics["recall_sensitivity"],
        label="Sensitivity"
    )

    plt.plot(
        threshold_metrics["threshold"],
        threshold_metrics["specificity"],
        label="Specificity"
    )

    plt.plot(
        threshold_metrics["threshold"],
        threshold_metrics["gmean"],
        label="G-mean"
    )

    for _, row in optimal_thresholds.iterrows():
        plt.axvline(
            row["threshold"],
            linestyle="--",
            alpha=0.6
        )

    plt.title("Wpływ progu odcięcia na metryki klasyfikacji")
    plt.xlabel("Próg odcięcia")
    plt.ylabel("Wartość metryki")
    plt.legend()
    plt.tight_layout()

    output_path = figures_dir / "metrics_vs_threshold.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrices_for_thresholds(y_true, y_score, optimal_thresholds, output_dir):
    rows = []

    for _, row in optimal_thresholds.iterrows():
        threshold = row["threshold"]
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        rows.append({
            "criterion": row["criterion"],
            "threshold": threshold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    confusion_df = pd.DataFrame(rows)

    output_path = output_dir / "confusion_matrices_selected_thresholds.csv"
    confusion_df.to_csv(output_path, index=False, encoding="utf-8")

    return confusion_df


def plot_confusion_matrices_for_thresholds(confusion_df, figures_dir):
    n_matrices = len(confusion_df)

    fig, axes = plt.subplots(
        1,
        n_matrices,
        figsize=(4 * n_matrices, 4)
    )

    if n_matrices == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, confusion_df.iterrows()):
        matrix = np.array([
            [row["tn"], row["fp"]],
            [row["fn"], row["tp"]],
        ])

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=[NEGATIVE_CLASS, POSITIVE_CLASS],
            yticklabels=[NEGATIVE_CLASS, POSITIVE_CLASS],
            ax=ax
        )

        ax.set_title(
            f"{row['criterion']}\nthreshold = {row['threshold']:.3f}",
            fontsize=9
        )
        ax.set_xlabel("Klasa przewidywana")
        ax.set_ylabel("Klasa rzeczywista")

    plt.tight_layout()

    output_path = figures_dir / "confusion_matrices_selected_thresholds.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    output_dir = STATISTICAL_TESTS_DIR / "logistic_regression_roc_binary"
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_binary_predictions()

    y_true = df["y_true"].values
    y_score = df["y_score"].values

    df_with_lr, coefficients, logistic_model = fit_logistic_regression(df)

    threshold_metrics = build_threshold_grid(y_true, y_score)
    optimal_thresholds = select_optimal_thresholds(threshold_metrics)

    auc_value = roc_auc_score(y_true, y_score)

    summary = pd.DataFrame([{
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "score_column": SCORE_COL,
        "n_samples": len(df),
        "auc": auc_value,
    }])

    confusion_matrices = save_confusion_matrices_for_thresholds(
        y_true=y_true,
        y_score=y_score,
        optimal_thresholds=optimal_thresholds,
        output_dir=output_dir
    )

    plot_confusion_matrices_for_thresholds(
        confusion_df=confusion_matrices,
        figures_dir=figures_dir
    )

    summary.to_csv(
        output_dir / "roc_summary_binary.csv",
        index=False,
        encoding="utf-8"
    )

    coefficients.to_csv(
        output_dir / "logistic_regression_coefficients.csv",
        index=False,
        encoding="utf-8"
    )

    threshold_metrics.to_csv(
        output_dir / "threshold_metrics_grid.csv",
        index=False,
        encoding="utf-8"
    )

    optimal_thresholds.to_csv(
        output_dir / "optimal_thresholds_binary.csv",
        index=False,
        encoding="utf-8"
    )

    df_with_lr.to_csv(
        output_dir / "binary_predictions_with_logistic_regression_probability.csv",
        index=False,
        encoding="utf-8"
    )

    plot_logistic_regression_curve(
        df=df_with_lr,
        model=logistic_model,
        figures_dir=figures_dir
    )

    logistic_threshold_metrics = plot_logistic_threshold_effect(
        df=df_with_lr,
        figures_dir=figures_dir
    )

    logistic_threshold_metrics.to_csv(
        output_dir / "logistic_regression_probability_threshold_metrics.csv",
        index=False,
        encoding="utf-8"
    )

    plot_roc_curve_with_thresholds(
        y_true=y_true,
        y_score=y_score,
        optimal_thresholds=optimal_thresholds,
        figures_dir=figures_dir
    )

    plot_metrics_vs_threshold(
        threshold_metrics=threshold_metrics,
        optimal_thresholds=optimal_thresholds,
        figures_dir=figures_dir
    )

    print("\nAnaliza regresji logistycznej i ROC dla modelu binarnego")
    print("Liczba próbek:", len(df))
    print("AUC:", auc_value)

    print("\nWspółczynniki regresji logistycznej")
    print(coefficients)

    print("\nWybrane progi odcięcia")
    print(optimal_thresholds[[
        "criterion",
        "threshold",
        "balanced_accuracy",
        "recall_sensitivity",
        "specificity",
        "fpr",
        "gmean",
        "youden_j",
    ]])


if __name__ == "__main__":
    main()