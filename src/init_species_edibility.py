'''
Informacja o wykorzystaniu AI:
Ten plik został opracowany przy wsparciu asystenta
ChatGPT (OpenAI, modele GPT-5 oraz GPT-5.5).
'''

import pandas as pd

from config import METADATA_DIR


def main():
    class_mapping_path = METADATA_DIR / "class_to_idx.csv"
    output_path = METADATA_DIR / "species_edibility.csv"

    df = pd.read_csv(class_mapping_path)

    edibility_df = pd.DataFrame({
        "species_label": df["species_label"],
        "edibility_label": "",
        "source": "",
        "notes": ""
    })

    edibility_df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()