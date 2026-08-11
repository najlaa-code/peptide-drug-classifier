
"""
Pipeline A mRMR

Uses mRMR (Minimum Redundancy Maximum Relevance) to pick features that are strongly related to the target 
(log10hc50 or log10mic) but not too redundant with each other. Only fits on the train split.

Notes:
- Zero-variance columns get dropped from the training features before mRMR runs.
- MID method is used (mutual information)
- mRMR is very slow, it takes about 2 hours to run on the full descriptor set.

Input:
- CSV from Script 1 (feature extraction), needs a Split column ("train"/"test") and the meta columns 
(SequenceIndex, Sequence, Class, log10hc50, log10mic, Split)
- --target: log10hc50 or log10mic (required)
- --max_features: how many features to keep (default 400)

Output:
- CSV with the meta columns + the selected features, for all rows (train and test), written to --output
- Printed: row/column counts, zero-variance columns dropped, train/test sizes, number of features selected, 
mRMR runtime, top 10 features by mutual information
"""

import argparse
import sys
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.selection import MRMR
import time
from sklearn.feature_selection import mutual_info_regression

META_COLUMNS = ["SequenceIndex", "Sequence", "Class", "log10hc50", "log10mic", "Split"]

def main(input_path: str, output_path:str, max_features:int, target: str):
    print("Loading data")
    try:
        df = pd.read_csv(input_path)
    except Exception:
        print(f"Error: the file '{input_path}' could not be loaded.")
        sys.exit(1)
    print(f"Data is loaded. Rows: {len(df)} and columns: {len(df.columns)}.")
    missing_columns = [column for column in META_COLUMNS if column not in df.columns]
    if missing_columns:
        print(f"Error: the columns '{missing_columns}' are missing from the input file.")
        sys.exit(1)

    feature_columns = [column for column in df.columns if column not in META_COLUMNS]

    # splitting
    train_mask = df["Split"] == "train"
    test_mask = df["Split"]== "test"
    print("Test/train split")
    print(f"test rows: {test_mask.sum()}")

    x_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, target]
    # Drop columns with 0 variance values
    zero_variance_columns = [column for column in feature_columns if x_train[column].std()==0]
    print(f"Dropping zero-variance columns. Length: {len(zero_variance_columns)}.")
    if zero_variance_columns:
        #x = x.drop(columns=zero_variance_columns)
        x_train = x_train.drop(columns=zero_variance_columns)

    feature_columns_kept= list(x_train.columns)
    maximum_features = min(max_features, len(feature_columns_kept))

    print(f"Afterdropping zero-variance columns. Length: {len(x_train.columns)}.")
    print(f"Train size: {len(x_train)}, Test size: {test_mask.sum()}")
    sel = MRMR(
        method="MID", # using mutual information
        regression=True, #continuous targets
        max_features=maximum_features,
    )
    print("Running mRMR now.")
    # ---debugging---
    t0 = time.time()
    sel.fit(x_train, y_train)
    elapsed = time.time() - t0
    print(f"mRMR fit completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes).")
    # ---debugging end---

    selected = [f for f in feature_columns_kept if f not in sel.features_to_drop_]
    print(f"Selected {len(selected)} features.")

    print("Ranking the features")
    mi_scores = mutual_info_regression(x_train[selected], y_train, random_state=42)
    mi_ranking = pd.Series(mi_scores, index=selected).sort_values(ascending=False)
    print("Top 10 features by relevance:")
    print(mi_ranking.head(10))

    # apply the same transformations to all rows
    df_out= df[[column for column in META_COLUMNS if column in df.columns]].copy()
    df_out = pd.concat([df_out, df[selected].reset_index(drop=True)], axis=1)
    if not output_path.endswith(".csv"):
        output_path = output_path.rsplit(".", 1)[0] + ".csv"
    print("Saving output")
    df_out.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP Feature Selection - Pipeline A (mRMR)")
    parser.add_argument("--input", "-i", default="features_raw.csv", help="Input CSV from Script 1")
    parser.add_argument("--output", "-o", default="features_mrmr_pipeline_A.csv", help="Output CSV")
    parser.add_argument("--max_features", "-n", type=int, default=400, help="Number of features to select")
    parser.add_argument("--target", "-t", choices=["log10hc50", "log10mic"], required=True)
    args = parser.parse_args()
    try:
        main(args.input, args.output, args.max_features, args.target)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
