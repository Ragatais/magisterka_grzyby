'''
Informacja o wykorzystaniu AI:
Ten plik został opracowany przy wsparciu asystenta
ChatGPT (OpenAI, modele GPT-5 oraz GPT-5.5).
'''

import pandas as pd

from config import SPLITS_DIR, METADATA_DIR


def load_edibility_mapping():
    path = METADATA_DIR / "species_edibility.csv"

    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    return df


def add_edibility(split_df, edibility_df):
    merged = split_df.merge(
        edibility_df[["species_label", "edibility_label"]],
        on="species_label",
        how="left"
    )

    return merged


def process_split(split_name, edibility_df):
    input_path = SPLITS_DIR / f"{split_name}.csv"
    output_path = SPLITS_DIR / f"{split_name}_with_edibility.csv"

    df = pd.read_csv(input_path)

    df = add_edibility(df, edibility_df)

    df.to_csv(output_path, index=False, encoding="utf-8")



def main():
    edibility_df = load_edibility_mapping()

    for split in ["train", "val", "test"]:
        process_split(split, edibility_df)



if __name__ == "__main__":
    main()