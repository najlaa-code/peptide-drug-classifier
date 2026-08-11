"""
Pipeline A Train/Test Split Script

Adds a "Split" column ("train"/"test") to the dataset, one single split used by everything downstream in 
Pipeline A (mRMR, LASSO, SVM, etc. all read this same column so the split stays consistent across the whole 
pipeline).

"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
def main(input_path, output_path, test_size, random_state):
    df = pd.read_csv(input_path)
    # stratify = None # if we do a random split
    stratify = df["Class"] if "Class" in df.columns and df["Class"].nunique() < 20 else None #stratify: to preserve the class proportions relative to the class imbalance
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=random_state, stratify=stratify
    )
    df["Split"] = "train"
    df.loc[test_idx, "Split"] = "test"
    print(f"train: {(df['Split'] == 'train').sum()}, test: {(df['Split'] == 'test').sum()}")
    df.to_csv(output_path, index=False)
    print(f"File saved in this path: {output_path}")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="features_raw_with_targets.csv")
    p.add_argument("--output", "-o", default="features_raw_prepped_A.csv")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()
    main(args.input, args.output, args.test_size, args.random_state)
