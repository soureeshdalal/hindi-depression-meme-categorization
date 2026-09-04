#!/usr/bin/env python3
"""
Merge unimodal and multimodal predictions into a single CSV for unified F1 calculation.
Combines all 8 models (3 unimodal + 5 multimodal) into one master predictions file.
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge unimodal and multimodal predictions"
    )
    parser.add_argument(
        "--unimodal",
        type=str,
        default="unimodal/MentalBERT_predictions.csv",
        help="Path to one unimodal model CSV (e.g. unimodal/<Model>_predictions.csv)",
    )
    parser.add_argument(
        "--multimodal",
        type=str,
        default="multimodal/PALO/PALO_predictions.csv",
        help="Path to one multimodal model CSV (e.g. under multimodal/<Model>/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_predictions.csv",
        help="Output merged CSV path",
    )
    args = parser.parse_args()

    print("=== MERGING PREDICTIONS ===")
    print(f"Loading unimodal predictions from {args.unimodal}...")
    df_uni = pd.read_csv(args.unimodal)
    print(f"  Found {len(df_uni)} rows")

    print(f"Loading multimodal predictions from {args.multimodal}...")
    df_multi = pd.read_csv(args.multimodal)
    print(f"  Found {len(df_multi)} rows")

    # Merge on image_id and category
    print("Merging datasets...")
    df_merged = pd.merge(
        df_uni,
        df_multi,
        on=["image_id", "category"],
        how="outer",
        suffixes=("_uni", "_multi"),
    )

    # Handle human_label columns (prefer non-empty values)
    if "human_label_uni" in df_merged.columns and "human_label_multi" in df_merged.columns:
        df_merged["human_label"] = df_merged["human_label_uni"].fillna(
            df_merged["human_label_multi"]
        )
        # If both are present, prefer the one that's not empty
        mask = (df_merged["human_label_uni"].astype(str).str.strip() != "") & (
            df_merged["human_label_uni"].notna()
        )
        df_merged.loc[mask, "human_label"] = df_merged.loc[mask, "human_label_uni"]
        df_merged = df_merged.drop(columns=["human_label_uni", "human_label_multi"])
    elif "human_label_uni" in df_merged.columns:
        df_merged = df_merged.rename(columns={"human_label_uni": "human_label"})
    elif "human_label_multi" in df_merged.columns:
        df_merged = df_merged.rename(columns={"human_label_multi": "human_label"})

    # Reorder columns: image_id, category, ocr_text, all model _pred/_raw columns, human_label
    base_cols = ["image_id", "category"]
    if "ocr_text" in df_merged.columns:
        base_cols.append("ocr_text")
    # Keep all columns that look like model predictions (Model_pred, Model_raw)
    model_cols = [c for c in df_merged.columns if c not in base_cols and c != "human_label"
                  and (c.endswith("_pred") or c.endswith("_raw"))]
    col_order = base_cols + sorted(model_cols)
    if "human_label" in df_merged.columns:
        col_order.append("human_label")
    df_merged = df_merged[[c for c in col_order if c in df_merged.columns]]

    # Save merged CSV
    print(f"Saving merged predictions to {args.output}...")
    df_merged.to_csv(args.output, index=False)
    print(f"Done. {len(df_merged)} rows saved.\n")

    n_models = len([c for c in df_merged.columns if c.endswith("_pred")])
    print("=== SUMMARY ===")
    print(f"Total images: {len(df_merged)}")
    print(f"Model prediction columns: {n_models}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
