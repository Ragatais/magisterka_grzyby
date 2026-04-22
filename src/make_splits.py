from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_DIR,
    SPLITS_DIR,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
    IMAGE_EXTENSIONS,
)


def collect_image_paths(data_dir: Path) -> pd.DataFrame:
    records = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        species_label = class_dir.name

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_path = image_path.relative_to(data_dir).as_posix()

            records.append(
                {
                    "relative_path": relative_path,
                    "species_label": species_label,
                }
            )

    df = pd.DataFrame(records)

    return df


def make_stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) > 1e-8:
        raise ValueError("TRAIN_SIZE + VAL_SIZE + TEST_SIZE musi dawać 1.0")

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["species_label"],
        random_state=RANDOM_STATE,
    )

    val_relative_size = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["species_label"],
        random_state=RANDOM_STATE,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def add_split_column(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):

    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(output_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(output_dir / "all_splits.csv", index=False, encoding="utf-8")


def print_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\nPodsumowanie")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    print(f"Razem: {len(train_df) + len(val_df) + len(test_df)}")

    print("\nClasses")
    print(f"Train: {train_df['species_label'].nunique()}")
    print(f"Val:   {val_df['species_label'].nunique()}")
    print(f"Test:  {test_df['species_label'].nunique()}")


def main():
    df = collect_image_paths(RAW_DATA_DIR)

    print(f"Znaleziono {len(df)} obrazów.")
    print(f"Liczba klas: {df['species_label'].nunique()}")

    train_df, val_df, test_df = make_stratified_split(df)
    train_df, val_df, test_df = add_split_column(train_df, val_df, test_df)

    save_splits(train_df, val_df, test_df, SPLITS_DIR)
    print_summary(train_df, val_df, test_df)


if __name__ == "__main__":
    main()