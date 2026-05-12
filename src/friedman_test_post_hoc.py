import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, rankdata, studentized_range

from config import METRICS_DIR, STATISTICAL_TESTS_DIR


METHOD_COLUMNS = [
    "binary_correct",
    "species_derived_correct",
    "hybrid_compromise_correct",
]

METHOD_LABELS = {
    "binary_correct": "Binary",
    "species_derived_correct": "Species-derived",
    "hybrid_compromise_correct": "Hybrid compromise",
}

ALPHA = 0.05


def load_comparison_data():
    binary_path = METRICS_DIR / "test_predictions_binary.csv"
    hybrid_path = METRICS_DIR / "hybrid_test_predictions_best_compromise.csv"

    binary = pd.read_csv(binary_path)
    hybrid = pd.read_csv(hybrid_path)

    binary = binary[
        ["relative_path", "true_label", "pred_label", "correct"]
    ].rename(columns={
        "true_label": "true_label_binary",
        "pred_label": "binary_pred_label",
        "correct": "binary_correct",
    })

    hybrid = hybrid[
        [
            "relative_path",
            "true_label",
            "final_decision",
            "edibility_from_species",
            "rejected",
        ]
    ].rename(columns={
        "true_label": "true_label_hybrid",
        "final_decision": "hybrid_pred_label",
    })

    merged = binary.merge(hybrid, on="relative_path", how="inner")

    merged = merged[merged["rejected"] == False].reset_index(drop=True)

    merged["binary_correct"] = merged["binary_correct"].astype(bool)

    merged["species_derived_correct"] = (
        merged["true_label_hybrid"] == merged["edibility_from_species"]
    )

    merged["hybrid_compromise_correct"] = (
        merged["true_label_hybrid"] == merged["hybrid_pred_label"]
    )

    return merged


def compute_ranks(df):
    values = df[METHOD_COLUMNS].astype(int).to_numpy()
    #-row, im lepsza metoda tym nizszaq ranga
    ranks = np.array([
        rankdata(-row, method="average")
        for row in values
    ])

    return ranks


def compute_mean_ranks(df):
    ranks = compute_ranks(df)
    mean_ranks = ranks.mean(axis=0)

    return pd.DataFrame({
        "method": METHOD_COLUMNS,
        "method_label": [METHOD_LABELS[m] for m in METHOD_COLUMNS],
        "mean_rank": mean_ranks,
    })


def run_friedman_test(df):
    statistic, p_value = friedmanchisquare(
        df["binary_correct"].astype(int),
        df["species_derived_correct"].astype(int),
        df["hybrid_compromise_correct"].astype(int),
    )

    return statistic, p_value


def nemenyi_post_hoc(mean_ranks_df, n_samples):
    """
    Post-hoc Nemenyi po teście Friedmana.
    p-value z rozkładu studentized range
    """
    k = len(mean_ranks_df)
    denominator = np.sqrt(k * (k + 1) / (6 * n_samples))

    results = []

    for i in range(k):
        for j in range(i + 1, k):
            method_i = mean_ranks_df.iloc[i]["method"]
            method_j = mean_ranks_df.iloc[j]["method"]

            label_i = mean_ranks_df.iloc[i]["method_label"]
            label_j = mean_ranks_df.iloc[j]["method_label"]

            rank_i = mean_ranks_df.iloc[i]["mean_rank"]
            rank_j = mean_ranks_df.iloc[j]["mean_rank"]

            rank_diff = abs(rank_i - rank_j)
            q_stat = rank_diff / denominator

            p_value = float(studentized_range.sf(q_stat, k, np.inf))

            results.append({
                "method_1": method_i,
                "method_2": method_j,
                "method_1_label": label_i,
                "method_2_label": label_j,
                "mean_rank_1": rank_i,
                "mean_rank_2": rank_j,
                "rank_difference": rank_diff,
                "q_statistic": q_stat,
                "p_value": p_value,
                "alpha": ALPHA,
                "significant": bool(p_value < ALPHA),
            })

    return pd.DataFrame(results)


def compute_critical_difference(n_samples, n_methods, alpha=ALPHA):
    """
    CD dla testu Nemenyi
    q_alpha z rozkładu studentized range.
    """
    q_alpha = studentized_range.ppf(1 - alpha, n_methods, np.inf)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_samples))

    return float(cd), float(q_alpha)


def plot_mean_ranks(mean_ranks_df, save_path):
    df = mean_ranks_df.sort_values("mean_rank", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(df["method_label"], df["mean_rank"])

    plt.xlabel("Mean rank")
    plt.ylabel("Method")
    plt.title("Mean ranks after Friedman test")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_critical_difference(mean_ranks_df, cd, save_path):
    df = mean_ranks_df.sort_values("mean_rank", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(11, 5))

    y_points = np.zeros(len(df))

    plt.scatter(df["mean_rank"], y_points, s=90)

    label_offsets = {
        "Hybrid compromise": 0.12,
        "Species-derived": 0.22,
        "Binary": 0.12,
    }

    for _, row in df.iterrows():
        label = row["method_label"]
        y_offset = label_offsets.get(label, 0.12)

        plt.text(
            row["mean_rank"],
            y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=10
        )

    min_rank = df["mean_rank"].min()
    max_rank = df["mean_rank"].max()

    cd_start = min_rank
    cd_end = min_rank + cd

    plt.plot([cd_start, cd_end], [-0.18, -0.18], linewidth=3)
    plt.text(
        (cd_start + cd_end) / 2,
        -0.25,
        f"CD = {cd:.4f}",
        ha="center",
        va="top",
        fontsize=10
    )

    plt.yticks([])
    plt.xlabel("Mean rank (lower is better)")
    plt.title("Critical Difference Diagram - Nemenyi post-hoc")

    padding = max(cd, 0.08)
    plt.xlim(min_rank - padding, max_rank + padding)
    plt.ylim(-0.35, 0.35)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    STATISTICAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_comparison_data()

    comparison_path = STATISTICAL_TESTS_DIR / "friedman_input_data.csv"
    df.to_csv(comparison_path, index=False, encoding="utf-8")

    statistic, p_value = run_friedman_test(df)

    mean_ranks_df = compute_mean_ranks(df)
    mean_ranks_path = STATISTICAL_TESTS_DIR / "friedman_mean_ranks.csv"
    mean_ranks_df.to_csv(mean_ranks_path, index=False, encoding="utf-8")

    friedman_result = {
        "test": "Friedman",
        "n_samples": len(df),
        "n_methods": len(METHOD_COLUMNS),
        "methods": ", ".join(METHOD_COLUMNS),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": ALPHA,
        "reject_H0": bool(p_value < ALPHA),
        "interpretation": (
            "Reject H0: not all mean ranks are equal; post-hoc analysis is justified."
            if p_value < ALPHA
            else "Do not reject H0: post-hoc analysis is not justified."
        ),
    }

    friedman_result_df = pd.DataFrame([friedman_result])
    friedman_result_path = STATISTICAL_TESTS_DIR / "friedman_test_results.csv"
    friedman_result_df.to_csv(friedman_result_path, index=False, encoding="utf-8")

    print("\nFriedman test results")
    print(friedman_result_df)

    print("\nmean ranks")
    print(mean_ranks_df)

    if p_value < ALPHA:
        nemenyi_df = nemenyi_post_hoc(mean_ranks_df, n_samples=len(df))
        nemenyi_path = STATISTICAL_TESTS_DIR / "nemenyi_post_hoc_results.csv"
        nemenyi_df.to_csv(nemenyi_path, index=False, encoding="utf-8")

        cd, q_alpha = compute_critical_difference(
            n_samples=len(df),
            n_methods=len(METHOD_COLUMNS),
            alpha=ALPHA
        )

        cd_result = pd.DataFrame([{
            "alpha": ALPHA,
            "n_samples": len(df),
            "n_methods": len(METHOD_COLUMNS),
            "q_alpha": q_alpha,
            "critical_difference": cd,
        }])

        cd_path = STATISTICAL_TESTS_DIR / "nemenyi_critical_difference.csv"
        cd_result.to_csv(cd_path, index=False, encoding="utf-8")

        plot_mean_ranks(
            mean_ranks_df,
            STATISTICAL_TESTS_DIR / "friedman_mean_ranks_plot.png"
        )

        plot_critical_difference(
            mean_ranks_df,
            cd,
            STATISTICAL_TESTS_DIR / "nemenyi_critical_difference_plot.png"
        )

        print("\nNemenyi post-oc results")
        print(nemenyi_df)

        print("\ncritical difference")
        print(cd_result)

        print(f"\nZapisano wyniki post-hoc: {nemenyi_path}")
        print(f"Zapisano CD: {cd_path}")
    else:
        print("\nNie wykonano post-hoc, ponieważ nie odrzucono H0 w teście Friedmana.")



if __name__ == "__main__":
    main()