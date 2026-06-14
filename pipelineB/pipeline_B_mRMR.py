
"""
Feature selection script 3: mRMR
- finds features with high relevance to the drug class
- finds features wtih low redundancy with each other
input: since this is pipeline A, features from feature extraction propy3
output: features_mrmr.csv
"""

import argparse
import sys
import traceback
import pandas as pd
from feature_engine.selection import MRMR
import time
from sklearn.feature_selection import mutual_info_regression

#META_COLUMNS = ["SequenceIndex", "Sequence", "Class"]
META_COLUMNS = ["SequenceIndex", "Sequence", "Class", "log10hc50", "log10mic"]


def main(input_path: str, output_path:str, max_features:int, target: str):
    print("Debugging 1: loading data or not")
    try:
        df = pd.read_csv(input_path)
    except Exception:
        print(f"Error: the file '{input_path}' could not be loaded.")
        sys.exit(1)
    print(f"Debugging 2: data is loaded. Rows: {len(df)} and columns: {len(df.columns)}.")
    missing_columns = [column for column in META_COLUMNS if column not in df.columns]
    if missing_columns:
        print(f"Error: the columns '{missing_columns}' are missing from the input file.")
        sys.exit(1)

    feature_columns = [column for column in df.columns if column not in META_COLUMNS]

    # Pipeline B: no splitting (the whole dataset is used)
    print("Debugging 3: pipeline B")
    #print(f"test rows: {test_mask.sum()}")
    x = df[feature_columns]
    y = df[target]

    # Drop columns with 0 variance values
    zero_variance_columns = [column for column in feature_columns if x[column].std() == 0]
    print(f"Debugging 4: dropping zero-variance columns. Length: {len(zero_variance_columns)}.")
    if zero_variance_columns:
        x = x.drop(columns=zero_variance_columns)
        #x_train = x_train.drop(columns=zero_variance_columns)

    feature_columns_kept= list(x.columns)
    maximum_features = min(max_features, len(feature_columns_kept))

    print(f"Debugging 5: afterdropping zero-variance columns. Length: {len(x.columns)}.")
    #maximum_features = min(max_features, len(x.columns)) # to remove maybe? this is for debugging to cap the max_features to the available features, as 1547 bugs and comes down to 1528
    #x_train, x_test, y_train, y_test = train_test_split(
    #    x, y, test_size = 0.2, random_state = 42
    #)
    print(f"Debugging 6: Full dataset size: {len(x)}")
    sel = MRMR(
        method="MID",
        #method="FCQ",
        regression=True, #changing due to continuous targets
        #max_features=max_features,
        max_features=maximum_features,
    )
    print("Running mRMR now.")
    # ---debugging---
    t0 = time.time()
    sel.fit(x, y)
    elapsed = time.time() - t0
    print(f"mRMR fit completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes).")
    # ---debugging end---

    selected = [f for f in feature_columns_kept if f not in sel.features_to_drop_]
    print(f"Selected {len(selected)} features.")

    print("Debugging 7/2: ranking the features")
    mi_scores = mutual_info_regression(x[selected], y, random_state=42)
    mi_ranking = pd.Series(mi_scores, index=selected).sort_values(ascending=False)
    print("Top 10 features by relevance:")
    print(mi_ranking.head(10))

    # apply the same transformations to all rows
    df_out= df[[column for column in META_COLUMNS if column in df.columns]].copy()
    df_out = pd.concat([df_out, df[selected].reset_index(drop=True)], axis=1)

    if not output_path.endswith(".csv"):
        output_path = output_path.rsplit(".", 1)[0] + ".csv"
    print("Debugging 7: saving output")
    df_out.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP Feature Selection - Pipeline B (mRMR)")
    parser.add_argument("--input", "-i", default="features_raw.csv", help="Input CSV from Script 1")
    parser.add_argument("--output", "-o", default="features_mrmr_pipeline_B.csv", help="Output CSV")
    parser.add_argument("--max_features", "-n", type=int, default=400, help="Number of features to select")
    parser.add_argument("--target", "-t", choices=["log10hc50", "log10mic"], required=True)
    args = parser.parse_args()
    try:
        main(args.input, args.output, args.max_features, args.target)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
