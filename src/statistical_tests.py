import numpy as np
import pandas as pd

from scipy.stats import binomtest, chi2

from config import METRICS_DIR, STATISTICAL_TESTS_DIR


def mcnemar_test(df, model_a_col, model_b_col):
    # dla dwóch sparowanych klasyfikatorów


    a_correct = df[model_a_col].astype(bool)
    b_correct = df[model_b_col].astype(bool)

    both_correct = int(((a_correct == True) & (b_correct == True)).sum())
    both_wrong = int(((a_correct == False) & (b_correct == False)).sum())

    a_correct_b_wrong = int(((a_correct == True) & (b_correct == False)).sum())
    a_wrong_b_correct = int(((a_correct == False) & (b_correct == True)).sum())

    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant == 0:
        exact_p_value = 1.0
        chi2_statistic = 0.0
        chi2_p_value = 1.0
    else:
        exact_p_value = binomtest(
            k=min(a_correct_b_wrong, a_wrong_b_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue

        chi2_statistic = ((abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2) / discordant
        chi2_p_value = 1 - chi2.cdf(chi2_statistic, df=1)

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "model_a_correct_model_b_wrong": a_correct_b_wrong,
        "model_a_wrong_model_b_correct": a_wrong_b_correct,
        "discordant_pairs": discordant,
        "mcnemar_exact_p_value": exact_p_value,
        "mcnemar_chi2_statistic": chi2_statistic,
        "mcnemar_chi2_p_value": chi2_p_value,
    }


def compare_multiclass_baseline_vs_tuned():
    baseline_path = METRICS_DIR / "test_predictions_multiclass_baseline.csv"
    tuned_path = METRICS_DIR / "test_predictions_multiclass_tuned.csv"

    baseline = pd.read_csv(baseline_path)
    tuned = pd.read_csv(tuned_path)

    baseline = baseline[["relative_path", "correct"]].rename(
        columns={"correct": "baseline_correct"}
    )

    tuned = tuned[["relative_path", "correct"]].rename(
        columns={"correct": "tuned_correct"}
    )

    merged = baseline.merge(tuned, on="relative_path", how="inner")

    result = mcnemar_test(
        merged,
        model_a_col="baseline_correct",
        model_b_col="tuned_correct"
    )

    result.update({
        "comparison": "multiclass_baseline_vs_tuned",
        "model_a": "multiclass_baseline",
        "model_b": "multiclass_tuned",
        "n_samples": len(merged),
    })

    return result


def compare_binary_vs_hybrid_compromise():
    binary_path = METRICS_DIR / "test_predictions_binary.csv"
    hybrid_path = METRICS_DIR / "hybrid_test_predictions_best_compromise.csv"

    binary = pd.read_csv(binary_path)
    hybrid = pd.read_csv(hybrid_path)

    binary = binary[["relative_path", "true_label", "pred_label", "correct"]].rename(
        columns={
            "true_label": "true_label_binary",
            "pred_label": "pred_label_binary",
            "correct": "binary_correct"
        }
    )

    hybrid = hybrid[
        ["relative_path", "true_label", "final_decision", "rejected"]
    ].rename(
        columns={
            "true_label": "true_label_hybrid",
            "final_decision": "pred_label_hybrid"
        }
    )

    merged = binary.merge(hybrid, on="relative_path", how="inner")

    merged = merged[merged["rejected"] == False].reset_index(drop=True)

    merged["hybrid_correct"] = (
        merged["true_label_hybrid"] == merged["pred_label_hybrid"]
    )

    result = mcnemar_test(
        merged,
        model_a_col="binary_correct",
        model_b_col="hybrid_correct"
    )

    result.update({
        "comparison": "binary_vs_hybrid_compromise",
        "model_a": "binary",
        "model_b": "hybrid_compromise",
        "n_samples": len(merged),
    })

    return result


def main():
    STATISTICAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    results.append(compare_multiclass_baseline_vs_tuned())
    results.append(compare_binary_vs_hybrid_compromise())

    results_df = pd.DataFrame(results)

    output_path = STATISTICAL_TESTS_DIR / "statistical_tests_mcnemar.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nMcNemar test results")
    print(results_df)



if __name__ == "__main__":
    main()