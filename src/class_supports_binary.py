import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import SPLITS_DIR, METRICS_DIR, STATISTICAL_TESTS_DIR


SPLIT_FILES = {
    "train": "train_known_edibility.csv",
    "val": "val_known_edibility.csv",
    "test": "test_known_edibility.csv",
}

CLASS_ORDER = ["edible", "poisonous"]


def load_split_supports():
    results = []

    for split_name, file_name in SPLIT_FILES.items():
        path = SPLITS_DIR / file_name
        df = pd.read_csv(path)

        counts = df["edibility_label"].value_counts()

        for class_name in CLASS_ORDER:
            n = int(counts.get(class_name, 0))

            results.append({
                "split": split_name,
                "class": class_name,
                "support": n,
            })

    results_df = pd.DataFrame(results)

    total_per_split = results_df.groupby("split")["support"].transform("sum")
    results_df["percentage"] = results_df["support"] / total_per_split * 100

    return results_df


def load_test_prediction_supports():
    path = METRICS_DIR / "test_predictions_binary.csv"
    df = pd.read_csv(path)

    true_counts = df["true_label"].value_counts()
    pred_counts = df["pred_label"].value_counts()

    results = []

    for class_name in CLASS_ORDER:
        results.append({
            "distribution": "true",
            "class": class_name,
            "support": int(true_counts.get(class_name, 0)),
        })

        results.append({
            "distribution": "predicted",
            "class": class_name,
            "support": int(pred_counts.get(class_name, 0)),
        })

    results_df = pd.DataFrame(results)

    total_per_distribution = results_df.groupby("distribution")["support"].transform("sum")
    results_df["percentage"] = results_df["support"] / total_per_distribution * 100

    return results_df


def plot_split_supports(results_df, figures_dir):
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=results_df,
        x="split",
        y="support",
        hue="class",
        hue_order=CLASS_ORDER,
    )

    plt.title("Rozkład wsparć klas w splitach modelu binarnego")
    plt.xlabel("Split")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()

    output_path = figures_dir / "binary_class_supports_by_split.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_test_prediction_supports(results_df, figures_dir):
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=results_df,
        x="distribution",
        y="support",
        hue="class",
        hue_order=CLASS_ORDER,
    )

    plt.title("Porównanie rzeczywistego i przewidywanego rozkładu klas")
    plt.xlabel("Rozkład")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()

    output_path = figures_dir / "binary_true_vs_predicted_class_supports.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    output_dir = STATISTICAL_TESTS_DIR / "class_supports_binary"
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    split_supports = load_split_supports()
    prediction_supports = load_test_prediction_supports()

    split_supports_path = output_dir / "binary_class_supports_by_split.csv"
    prediction_supports_path = output_dir / "binary_true_vs_predicted_class_supports.csv"

    split_supports.to_csv(split_supports_path, index=False, encoding="utf-8")
    prediction_supports.to_csv(prediction_supports_path, index=False, encoding="utf-8")

    plot_split_supports(split_supports, figures_dir)
    plot_test_prediction_supports(prediction_supports, figures_dir)

    print("\nWsparcia klas w splitach modelu binarnego")
    print(split_supports)

    print("\nPorównanie rzeczywistego i przewidywanego rozkładu klas na zbiorze testowym")
    print(prediction_supports)



if __name__ == "__main__":
    main()