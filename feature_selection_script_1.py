"""Feature extraction
include more text here
Requirement: propy3 installed
I feel like my for loops will crash... so I will have a bunch of warnings
"""

import argparse
import sys
import traceback
import pandas as pd
from tqdm.auto import tqdm
import importlib.resources
import types

pkg_resources = types.ModuleType("pkg_resources")
def _resource_filename(package, path):
    return str(importlib.resources.files(package).joinpath(path))
pkg_resources.resource_filename = _resource_filename
sys.modules["pkg_resources"] = pkg_resources

try:
    from propy import AAComposition, Autocorrelation, CTD, PseudoAAC, QuasiSequenceOrder
except ImportError as e:
    #print("Propy3 not found. Install it. https://github.com/MartinThoma/propy3")
    print(f"Import failed: {e}")
    sys.exit(1)
INPUT_SEQUENCE = "sequence" # based on the excel file column
INPUT_CLASS = "drug_class"

def safe_extract(func, sequence: str, Label: str):
    """Since there is 4810 sequences, this wrapper function avoids crashes and returns {} when the program fails. This is a ProPy3 function."""
    try:
        return func(sequence)
    except Exception as exc:
        print(f"This sequence: '{sequence}' failed for sequence: {exc}")
        return {}

def extract_descriptors(sequence: str):
    """This function will use all the 1529 ProPy3 descriptors on each sequence (4810 in total)
    Descriptor groups and approximate column counts:
        AAC          –   20   (amino acid composition)
        DPC          –  400   (dipeptide composition)
        TPC          – 8000   (tripeptide – EXCLUDED: too many columns)
        MoreauBroto  –  240   (normalized autocorrelation)
        Moran        –  240
        Geary        –  240
        CTD          –  147   (composition/transition/distribution)
        ConjointTriad–  343
        SOCNumber    –   60   (sequence-order coupling)
        QSODescriptor–  100
        PseAAC       –   50
        APseAAC      –   80
        ──────────────────
        Total        – ~1529  (TPC excluded to stay within column budget)"""
    features = {}

    # ---Amino Acids Composition--- https://propy3.readthedocs.io/en/latest/_modules/propy/AAComposition.html
    # AAC
    features.update(safe_extract(AAComposition.CalculateAAComposition, sequence, "AAC"))
    # DPC
    features.update(safe_extract(AAComposition.CalculateDipeptideComposition, sequence, "DPC"))

    # ---Autocorrelation--- https://propy3.readthedocs.io/en/latest/_modules/propy/Autocorrelation.html
    # Moreau-Broto Autocorrelation
    features.update(safe_extract(
        # protein sequence, AAProperty, AAPropertyName
        lambda params: Autocorrelation.CalculateNormalizedMoreauBrotoAuto(sequence), sequence, "NormalizedMoreauBroto"
    ))
    # Moran autocorrelation
    features.update(safe_extract(
        lambda params: Autocorrelation.CalculateMoranAuto(params,[1]*8), sequence, "Moran"
    ))
    # Geary autocorrelation
    features.update(safe_extract(
        lambda params: Autocorrelation.CalculateGearyAuto(params, [1]*8), sequence, "Geary"
    ))

    #---CTD-- (combines composition, transition, and distribution)
    features.update(safe_extract(
        CTD.CalculateCTD, sequence, "CTD"
    ))

    # ---Quasi-sequence order--- # From chapter 14,https://propy3.readthedocs.io/_/downloads/en/latest/pdf/
    # Sequence order coupling number
    features.update(safe_extract(
        lambda params: QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(params, maxlag=30), sequence, "SequenceOrder"
    ))
    # Quasi-sequence order descriptors
    features.update(safe_extract(
        lambda params: QuasiSequenceOrder.GetQuasiSequenceOrder(params), sequence, "QuasiSequence"
    ))

    #---Pseudo amino acid compostion
    # Pseudo amino acid composition
    features.update(safe_extract(
        lambda params: PseudoAAC.GetPseudoAAC(params, lamda=30, weight=0.05), sequence, "PseudoAAC"
    ))
    # Amphiphilic pseudo amino acid composition
    features.update(safe_extract(
        lambda params: PseudoAAC.GetAPseudoAAC(params, lamda=30, weight=0.05), sequence, "APseudoAAC"
    ))
    return features

# Main
def main(input_path: str, output_path:str) -> None:
    try:
        df_in = pd.read_excel(input_path)
    except Exception:
        print(f"Error: file '{input_path}' could not be loaded.")

    if INPUT_SEQUENCE not in df_in.columns:
        sys.exit(f"Error: no '{INPUT_SEQUENCE}' columns was found.")

    sequences = df_in[INPUT_SEQUENCE].astype(str).tolist()
    labels =  df_in[INPUT_CLASS].tolist() if INPUT_CLASS in df_in.columns else [None]*len(sequences)
    # extracting features
    rows: list[dict]=[]
    failed: list[int]=[]
    for index, (seq, label) in enumerate(tqdm(zip(sequences, labels), total = len(sequences), unit="seq")):
        seq_clean = seq.strip().upper()
        valid_AAsequence = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(seq_clean).issubset(valid_AAsequence):
            print(f"Error: the sequence '{seq}' is not valid.")
            failed.append(index)
            continue

    descriptors = extract_descriptors(seq_clean)
    descriptors["SequenceIndex"] = index
    descriptors["Sequence"] = seq_clean
    descriptors["Class"] = label
    rows.append(descriptors)
    # export it to CSV
    df_feat = pd.DataFrame(rows)
    meta_cols = ["SequenceIndex", "Sequence", "Class"]
    feature_cols = [col for col in df_feat.columns if col not in meta_cols]
    df_feat = df_feat[meta_cols + feature_cols]
    df_feat[feature_cols] = df_feat[feature_cols].fillna(0)
    if not output_path.endswith(".csv"):  # CHANGED
        output_path = output_path.rsplit(".", 1)[0] + ".csv"
    df_feat.to_csv(output_path, index=False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP Feature Extraction – Script 1")
    parser.add_argument("--input", "-i", default="dbaasp_grampa_hemolytik_hc50_mic.xlsx", help="Input file")
    parser.add_argument("--output", "-o", default="features_raw.csv", help="Output CSV file")
    args = parser.parse_args()
    try:
        main(args.input, args.output)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
