"""
Data loading utilities for Colab inference experiments.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_label(label_str) -> list[int]:
    """
    Parse label string/list into a 7-length integer vector.
    """
    if isinstance(label_str, list):
        vec = [int(x) for x in label_str]
        return vec if len(vec) == 7 else [0] * 7
    if pd.isna(label_str):
        return [0] * 7
    try:
        val = ast.literal_eval(str(label_str))
        if isinstance(val, (list, tuple)) and len(val) == 7:
            return [int(x) for x in val]
    except Exception:
        pass
    return [0] * 7


def load_metadata(metadata_path, explanations_path=None) -> pd.DataFrame:
    """
    Load metadata and optionally merge explanations.
    Returns dataframe with canonical columns.
    """
    df = pd.read_csv(metadata_path)
    required = ["image_id", "category", "ocr_text", "human_label", "image_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")

    if explanations_path:
        exp = pd.read_csv(explanations_path)
        if "image_id" not in exp.columns:
            raise ValueError("explanations.csv missing image_id")
        if "explanation_implicit" not in exp.columns:
            for cand in ("explanation", "explanation_direct", "implicit_explanation"):
                if cand in exp.columns:
                    exp = exp.rename(columns={cand: "explanation_implicit"})
                    break
        if "explanation_implicit" in exp.columns:
            exp = exp[["image_id", "explanation_implicit"]].drop_duplicates("image_id")
            df = df.merge(exp, on="image_id", how="left")
        else:
            df["explanation_implicit"] = ""
    else:
        df["explanation_implicit"] = ""

    df["human_label_vec"] = df["human_label"].map(parse_label)
    return df


def load_image(image_path):
    """
    Load image as RGB PIL image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    return Image.open(path).convert("RGB")


class InferenceDataset:
    """
    Simple indexable dataset returning image + row metadata.
    """

    def __init__(self, metadata_df: pd.DataFrame, image_dir: str | Path):
        self.df = metadata_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(self.image_dir / Path(row["image_path"]).name)
        return img, row.to_dict()
