import pandas as pd

from config import SPLITS_DIR


VALID_LABELS = {"edible", "poisonous"}


def clean_edibility_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    df["edibility_label"] = (
        df["edibility_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def filter_known_edibility(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["edibility_label"].isin(VALID_LABELS)].reset_index(drop=True)


def summarize_split(df_before: pd.DataFrame, df_after: pd.DataFrame, split_name: str):
    print(f"\n{split_name.upper()}:")
    print(f"Przed filtrowaniem: {len(df_before)}")
    print(f"Po filtrowaniu:     {len(df_after)}")
    print(f"Usunięto unknown:   {len(df_before) - len(df_after)}")

    print("\nLiczność etykiet jadalności:")
    print(df_after["edibility_label"].value_counts())

    print("\nLiczba gatunków po filtrowaniu:")
    print(df_after["species_label"].nunique())


def process_split(split_name: str):
    input_path = SPLITS_DIR / f"{split_name}_with_edibility.csv"
    output_path = SPLITS_DIR / f"{split_name}_known_edibility.csv"


    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = clean_edibility_labels(df)

    df_known = filter_known_edibility(df)


    df_known.to_csv(output_path, index=False, encoding="utf-8")

    summarize_split(df, df_known, split_name)


def main():
    for split_name in ["train", "val", "test"]:
        process_split(split_name)


if __name__ == "__main__":
    main()