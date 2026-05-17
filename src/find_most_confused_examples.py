from pathlib import Path
import shutil

import pandas as pd

from config import RAW_DATA_DIR, METRICS_DIR, OUTPUTS_DIR


TOP_N_PAIRS = 5
EXAMPLES_PER_PAIR = 3

ERROR_ANALYSIS_DIR = OUTPUTS_DIR / "error_analysis"
CONFUSED_EXAMPLES_DIR = ERROR_ANALYSIS_DIR / "most_confused_examples"


def safe_name(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def main():
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSED_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = METRICS_DIR / "test_predictions_multiclass_tuned.csv"

    df = pd.read_csv(predictions_path)

    required_columns = {"relative_path", "true_label", "pred_label", "correct"}
    missing_columns = required_columns - set(df.columns)


    errors = df[df["correct"] == False].copy()

    confused_pairs = (
        errors
        .groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    confused_pairs.to_csv(
        ERROR_ANALYSIS_DIR / "most_confused_pairs_multiclass_tuned.csv",
        index=False,
        encoding="utf-8"
    )

    top_pairs = confused_pairs.head(TOP_N_PAIRS)

    selected_examples = []

    for pair_idx, row in top_pairs.iterrows():
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        count = row["count"]

        pair_dir_name = (
            f"{pair_idx + 1:02d}_true_{safe_name(true_label)}"
            f"__pred_{safe_name(pred_label)}"
        )

        pair_dir = CONFUSED_EXAMPLES_DIR / pair_dir_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        pair_errors = errors[
            (errors["true_label"] == true_label)
            & (errors["pred_label"] == pred_label)
        ].head(EXAMPLES_PER_PAIR)

        for example_idx, example in pair_errors.iterrows():
            relative_path = example["relative_path"]
            source_path = RAW_DATA_DIR / relative_path

            extension = source_path.suffix
            destination_name = (
                f"example_{len(selected_examples) + 1:03d}"
                f"_true_{safe_name(true_label)}"
                f"_pred_{safe_name(pred_label)}"
                f"{extension}"
            )

            destination_path = pair_dir / destination_name

            shutil.copy2(source_path, destination_path)

            selected_examples.append({
                "pair_rank": pair_idx + 1,
                "true_label": true_label,
                "pred_label": pred_label,
                "pair_count": int(count),
                "original_relative_path": relative_path,
                "copied_path": str(destination_path.relative_to(OUTPUTS_DIR)),
            })

    pd.DataFrame(selected_examples).to_csv(
        ERROR_ANALYSIS_DIR / "selected_confused_examples_multiclass_tuned.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nZapisano ranking najczęściej mylonych par:")
    print(ERROR_ANALYSIS_DIR / "most_confused_pairs_multiclass_tuned.csv")

    print("\nSkopiowano przykładowe obrazy do:")
    print(CONFUSED_EXAMPLES_DIR)

    print("\nZapisano listę wybranych przykładów:")
    print(ERROR_ANALYSIS_DIR / "selected_confused_examples_multiclass_tuned.csv")

    print("\nTOP najczęściej mylonych par:")
    print(top_pairs)


if __name__ == "__main__":
    main()