"""
SVM for Pipeline B (HC50 + MIC combined)
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import json
import joblib
import shap

BASE_DIR = "/mnt/c/Users/najla/Downloads/SCOL 391 - Mansbach/CleaningDBsScripts/dbaasp_grampa_hemolytik_hc50_mic_ANALYSIS"
HC50_CSV = os.path.join(BASE_DIR, "lasso_pipeline_B_HC50_features.csv")
MIC_CSV  = os.path.join(BASE_DIR, "lasso_pipeline_B_MIC_features.csv")
OUT_CM   = os.path.join(BASE_DIR, "svm_confusion_matrix_pipeline_B_UNION.png")

LABEL_COL = "Class"
META_DROP = ["SequenceIndex", "Sequence", "Class", "log10hc50", "log10mic", "Split"]

print("Loading feature sets")
df_hc50 = pd.read_csv(HC50_CSV)
df_mic  = pd.read_csv(MIC_CSV)

meta_cols = [c for c in META_DROP if c in df_hc50.columns and c in df_mic.columns]
if "Split" not in meta_cols:
    raise ValueError("No Split column")

merge_keys = [c for c in ["SequenceIndex", "Sequence"] if c in meta_cols]
if not merge_keys:
    raise ValueError("No SequenceIndex/Sequence")
hc50_features = [c for c in df_hc50.columns if c not in meta_cols]
mic_features  = [c for c in df_mic.columns if c not in meta_cols]
overlap_features = sorted(set(hc50_features) & set(mic_features))
print(f"HC50-selected features #: {len(hc50_features)}")
print(f"MIC-selected features #:  {len(mic_features)}")
print(f"Overlapping features between HC-50 and MIC selected features: {len(overlap_features)}")

merged = df_hc50.merge(
    df_mic[merge_keys + mic_features],
    on=merge_keys,
    how="inner",
    suffixes=("", "_mic_dup"),
)
for feat in overlap_features:
    dup_col = f"{feat}_mic_dup"
    if dup_col in merged.columns:
        mismatch = ~np.isclose(merged[feat], merged[dup_col], equal_nan=True)
        if mismatch.any():
            raise AssertionError(f"Feature [{feat}] differs between HC50 and MIC LASSO. {mismatch.sum()} peptides")
        merged = merged.drop(columns=[dup_col])
union_features = sorted(set(hc50_features) | set(mic_features))
print(f"Union feature set: {len(union_features)} features "
      f"({len(hc50_features)} + {len(mic_features)} - {len(overlap_features)} overlap)")

if merged[LABEL_COL].isna().any() or len(merged) != len(df_hc50):
    raise AssertionError(f"Merge did not preserve the full peptide set: {len(merged)} rows after merge (total). Hc50: {len(df_hc50)}. ")

train_mask = merged["Split"] == "train"
test_mask = merged["Split"] == "test"
X_train = merged.loc[train_mask, union_features]
y_train = merged.loc[train_mask, LABEL_COL]
X_test  = merged.loc[test_mask, union_features]
y_test  = merged.loc[test_mask, LABEL_COL]
print(f"Training size: {len(X_train)} rows.")
print(f"Testing size: {len(X_test)} rows.")
print(f"Features: {X_train.shape[1]}")

#based on pipeline_B_hyperparameter_search numbers, hardcoded
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Classification results:")
print(classification_report(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring="f1_weighted")
print(f"Mean of cross-validation scores: {cv_scores.mean():.3f}")
print(f"Standard deviation of cross-validation scores: {cv_scores.std():.3f}")

cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="RdPu", values_format="d")
disp.im_.colorbar.set_label("Number of peptides", fontsize=13)
ax.set_xlabel("Predicted Class", fontsize=14)
ax.set_ylabel("True Class", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.set_title("SVM Confusion Matrix Pipeline B (HC50+MIC features)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_CM, dpi=150)
plt.close()
# SHAP
svm_probability = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)
svm_probability.fit(X_train, y_train)
K_BACKGROUND = 25
N_PER_CLASS  = 30
TOPN         = 15
feature_names = list(X_train.columns)
classes = list(svm_probability.classes_)
np.random.seed(42)
background = shap.kmeans(X_train.values, K_BACKGROUND)
print("debug 1: building explainer")
explainer = shap.KernelExplainer(svm_probability.predict_proba, background)
rng = np.random.RandomState(42)
explain_idx = []
for c in classes:
    idx_c = y_test.index[y_test == c].to_numpy()
    take  = min(N_PER_CLASS, len(idx_c))
    explain_idx += list(rng.choice(idx_c, size=take, replace=False))
X_explain = X_test.loc[explain_idx]

shap_values = explainer.shap_values(X_explain.values)
if isinstance(shap_values, list):
    sv_list = shap_values
else:
    sv_list = [shap_values[:, :, i] for i in range(len(classes))]

imp = pd.DataFrame(
    {c: pd.Series(np.abs(sv_list[i]).mean(axis=0), index=feature_names)
     for i, c in enumerate(classes)}
)
imp.to_csv(os.path.join(BASE_DIR, "shap_pipeline_B_union_with_shap_mean_abs.csv"))

def safe(s):
    return s.replace(" ", "_").replace(":", "").replace("/", "-")

for i, c in enumerate(classes):
    plt.figure()
    shap.summary_plot(sv_list[i], X_explain.values, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=TOPN)
    plt.title(f"SHAP mean |value| — {c} (Union HC50+MIC, Pipeline B)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"shap_bar_pipeline_B_UNION_{safe(c)}.png"), dpi=150)
    plt.close()

RARE = next(c for c in classes if "Both Active" in c)
MAJ = next(c for c in classes if "Inactive" in c)
rare_top = list(imp[RARE].sort_values(ascending=False).head(TOPN).index)
maj_top = list(imp[MAJ].sort_values(ascending=False).head(TOPN).index)
overlap = set(rare_top) & set(maj_top)
jacc = len(overlap) / len(set(rare_top) | set(maj_top))
print("\nDiagnostic report:")
print(f"Top {TOPN} descriptors for RARE class ({RARE}):")
print("  " + ", ".join(rare_top))
print(f"\nTop {TOPN} descriptors for MAJORITY class ({MAJ}):")
print("  " + ", ".join(maj_top))
print(f"\nShared descriptors: {len(overlap)}-Jaccard overlap: {jacc:.2f}")
print("  " + (", ".join(sorted(overlap)) if overlap else "(none)"))
