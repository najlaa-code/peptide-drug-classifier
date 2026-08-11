"""
Pipeline A Feature Scaling

Standardizes all feature columns with StandardScaler (z = (x - mean) / std). The scaler is fit on the train 
split only, then applied to the entire dataset (train + test).

Input:
    - CSV from Script 1 (feature extraction), needs a Split column
      ("train"/"test") and the meta columns (SequenceIndex, Sequence, Class,
      log10hc50, log10mic, Split)

Output:
    - Scaled CSV (meta columns + scaled features, all rows), written to
      --output
    - Printed: sequence count, feature count

"""
import sys
import argparse
import traceback
import pandas as pd
from sklearn.preprocessing import StandardScaler

META_COLUMNS = ["SequenceIndex", "Sequence", "Class", "log10hc50", "log10mic", "Split"]
def main(input_path: str, output_path: str):
    try:
        df = pd.read_csv(input_path)
        print("debug 1")
    except Exception:
        print(f"Error: file '{input_path}' could not be loaded.")
        sys.exit(1)
    missing = [columns for columns in META_COLUMNS if columns not in df.columns]
    if missing:
        print(f"Error: missing expected columns: {missing}")
        sys.exit(1)
    feature_columns = [columns for columns in df.columns if columns not in META_COLUMNS]
    if not feature_columns:
        print("Error: no columns were found.")
        sys.exit(1)
    print(f"Number of loaded sequences: '{len(df)}'.")
    print(f"Number of loaded features: '{len(feature_columns)}'.")

    #Scale
    train_mask = df["Split"] == "train"
    print(f"debug 2")
    scaler = StandardScaler()  # z = (x - mean) / std
    scaler.fit(df.loc[train_mask, feature_columns])
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.transform(df[feature_columns])
    df_scaled = df_scaled[META_COLUMNS + feature_columns]

    # Export
    if not output_path.endswith(".csv"): # force the output to be .csv
        output_path = output_path.rsplit(".",1)[0]+".csv"
    df_scaled.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP Feature Scaling - Pipeline A")
    parser.add_argument("--input", "-i", default="features_raw.csv", help="Input CSV from Script 1")
    parser.add_argument("--output", "-o", default="features_scaled.csv", help="Output scaled CSV")
    args = parser.parse_args()
    try:
        main(args.input, args.output)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
