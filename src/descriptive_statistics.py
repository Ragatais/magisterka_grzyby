import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import METRICS_DIR, STATISTICAL_TESTS_DIR


FEATURES = [
    "species_confidence",
    "binary_confidence",
]

CATEGORIES = [
    "edible",
    "poisonous",
    "reject",
]


def coefficient_of_variation(values):
    mean = values.mean()

    if mean == 0 or np.isnan(mean):
        return np.nan

    return values.std(ddof=1) / mean * 100


def get_mode(values):
    rounded_values = values.round(4)
    mode_values = rounded_values.mode()

    if mode_values.empty:
        return np.nan

    return mode_values.iloc[0]


def descriptive_statistics_for_feature(df, feature):
    results = []

    for category in CATEGORIES:
        values = df.loc[df["final_decision"] == category, feature].dropna()

        q1 = values.quantile(0.25)
        q2 = values.quantile(0.50)
        q3 = values.quantile(0.75)

        results.append({
            "feature": feature,
            "category": category,
            "n": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "mode": get_mode(values),
            "variance": values.var(ddof=1),
            "std": values.std(ddof=1),
            "Q1": q1,
            "Q2_median": q2,
            "Q3": q3,
            "IQR": q3 - q1,
            "coefficient_of_variation_percent": coefficient_of_variation(values),
        })

    return results


def plot_histogram(df, feature, output_dir):
    plt.figure(figsize=(8, 5))

    plt.hist(
        df[feature].dropna(),
        bins=30,
        edgecolor="black"
    )

    plt.title(f"Histogram rozkładu: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Liczba obserwacji")
    plt.tight_layout()

    output_path = output_dir / f"histogram_{feature}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_grouped_boxplot(df, feature, output_dir):
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="final_decision",
        y=feature,
        order=CATEGORIES,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        }
    )

    plt.title(f"Wykres pudełkowy: {feature} według decyzji końcowej")
    plt.xlabel("Decyzja końcowa")
    plt.ylabel(feature)
    plt.tight_layout()

    output_path = output_dir / f"boxplot_grouped_{feature}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_and_filter_hybrid_predictions():
    input_path = METRICS_DIR / "hybrid_test_predictions_best_constrained_coverage_0_70.csv"

    df = pd.read_csv(input_path)

    df = df.dropna(subset=FEATURES).copy()
    df = df[df["final_decision"].isin(CATEGORIES)].reset_index(drop=True)

    return df


def main():
    output_dir = STATISTICAL_TESTS_DIR / "descriptive_statistics"
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter_hybrid_predictions()

    print("\nDane wejściowe do statystyki opisowej")
    print("Liczba próbek po filtracji:", len(df))

    print("\nLiczności kategorii")
    print(df["final_decision"].value_counts())

    all_results = []

    for feature in FEATURES:
        all_results.extend(
            descriptive_statistics_for_feature(df, feature)
        )

        plot_histogram(
            df=df,
            feature=feature,
            output_dir=figures_dir
        )

        plot_grouped_boxplot(
            df=df,
            feature=feature,
            output_dir=figures_dir
        )

    results_df = pd.DataFrame(all_results)

    output_path = output_dir / "descriptive_statistics_hybrid_confidence.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    filtered_data_path = output_dir / "descriptive_statistics_filtered_data.csv"
    df.to_csv(filtered_data_path, index=False, encoding="utf-8")

    print("\nWyniki statystyki opisowej")
    print(results_df)



if __name__ == "__main__":
    main()