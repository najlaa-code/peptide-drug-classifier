"""
Pipeline A Step 0
Merges loc10hc50 / log10mic into the features file
"""
import argparse
import sys
import pandas as pd

def main(features_path, db_path, output_path):
    print("Debug 1: loading features")
    features = pd.read_csv(features_path)
    print("Debug 2: loading database")
    db = pd.read_csv(db_path)
    # normalize Sequence to sequence
    db = db.rename(columns={"sequence": "Sequence"})
    key =  "SequenceIndex" if ("SequenceIndex" in db.columns and "SequenceIndex" in features.columns) else "Sequence"
    needed = [key, "log10hc50", "log10mic"]
    missing = [c for c in needed if c not in db.columns]
    if missing:
        print(f"Error: cannot find {missing}.")
        sys.exit(1)
    merged = features.merge(db[needed], on=key, how="left")
    n_missing = merged["log10hc50"].isna().sum()
    if n_missing > 0:
        print(f"  dropping {n_missing} rows.")
        merged = merged.dropna(subset=["log10hc50", "log10mic"]).reset_index(drop=True)
    merged.to_csv(output_path, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", "-f", default="features_raw.csv")
    p.add_argument("--db", "-d", required=True)
    p.add_argument("--output", "-o", default="features_raw_merged.csv")
    args = p.parse_args()
    main(args.features, args.db, args.output)
