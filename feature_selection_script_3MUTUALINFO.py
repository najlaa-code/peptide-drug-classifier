"""
Feature selection script 3: mRMR
- finds features with high relevance to the drug class
- finds features wtih low redundancy with each other
input: features_scaled (from script 2)
output: features_mrmr.csv
"""

import argparse
import sys
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.selection import MRMR
#debugging
import time

META_COLUMNS = ["SequenceIndex", "Sequence", "Class"]

def main(input_path: str, output_path:str, max_features:int):
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
    x = df[feature_columns]
    y = df["Class"]
    print("Debugging 3: feature columns")

    # Drop columns with 0 variance values
    zero_variance_columns = [column for column in feature_columns if x[column].std()==0]
    print(f"Debugging 4: dropping zero-variance columns. Length: {len(zero_variance_columns)}.")
    if zero_variance_columns:
        x = x.drop(columns=zero_variance_columns)
    print(f"Debugging 5: afterdropping zero-variance columns. Length: {len(x.columns)}.")
    maximum_features = min(max_features, len(x.columns)) # to remove maybe? this is for debugging to cap the max_features to the available features, as 1547 bugs and comes down to 1528
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, random_state = 42, stratify=y
    )
    print(f"Debugging 6: Train size: {len(x_train)}, Test size: {len(x_test)}")
    sel = MRMR(
        method="MID",
        #method="FCQ",
        regression=False,
        #max_features=max_features,
        max_features=maximum_features,
    )
    print("Running mRMR now.")
    # ---debugging---
    # NEW
    t0 = time.time()
    sel.fit(x_train, y_train)
    # NEW
    elapsed = time.time() - t0
    print(f"mRMR fit completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes).")
    # ---debugging---


    selected = [f for f in x.columns if f not in sel.features_to_drop_]
    print(f"Selected {len(selected)} features:")
    print(selected)
    x_selected = x[selected]
    df_out = df[META_COLUMNS].copy()
    df_out = pd.concat([df_out, x_selected.reset_index(drop=True)], axis=1)
    if not output_path.endswith(".csv"):
        output_path = output_path.rsplit(".", 1)[0] + ".csv"
    print("Debugging 7: saving output")
    df_out.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP Feature Selection - Script 3 (mRMR)")
    parser.add_argument("--input", "-i", default="features_scaled.csv", help="Input CSV from Script 2")
    parser.add_argument("--output", "-o", default="features_mrmr.csv", help="Output CSV")
    parser.add_argument("--max_features", "-n", type=int, default=248, help="Number of features to select")
    args = parser.parse_args()
    try:
        main(args.input, args.output, args.max_features)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
