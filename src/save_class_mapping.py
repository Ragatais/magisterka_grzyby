import pandas as pd

from config import SPLITS_DIR, METADATA_DIR


def main():
    train_csv = SPLITS_DIR / "train.csv"

    df = pd.read_csv(train_csv)

    classes = sorted(df["species_label"].unique())

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    mapping_df = pd.DataFrame([
        {"species_label": k, "class_idx": v}
        for k, v in class_to_idx.items()
    ])


    output_path = METADATA_DIR / "class_to_idx.csv"
    mapping_df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()