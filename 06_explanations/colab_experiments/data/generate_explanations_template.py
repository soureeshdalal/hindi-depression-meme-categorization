"""
Template utility if explanations.csv is missing.

Fill `explanation_implicit` externally (LLM/annotation), then save as:
  colab_experiments/data/explanations.csv
with columns:
  - image_id
  - explanation_implicit
"""

from pathlib import Path

import pandas as pd


def main():
    base = Path(__file__).resolve().parent
    meta = base / "metadata.csv"
    out = base / "explanations_template.csv"
    df = pd.read_csv(meta)
    temp = pd.DataFrame(
        {
            "image_id": df["image_id"],
            "explanation_implicit": [""] * len(df),
        }
    )
    temp.to_csv(out, index=False)
    print(f"Wrote template: {out}")


if __name__ == "__main__":
    main()
