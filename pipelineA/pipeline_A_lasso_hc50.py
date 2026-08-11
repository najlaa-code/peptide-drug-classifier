"""
Pipeline A - LASSO Regression (HC50)

Takes mRMR selected features and runs LASSO on top (LassoCV, max_iter=10_000 since default 1000 wasn't enough given how many
features there are and it kept throwing convergence warnings). LASSO L1 penalty shrinks coefficients of 
less-informative features to 0, so whatever survives goes forward in the pipeline.

Notes:
- Only fit on the train split. Test stays untouched.

Input:
    - features_mRMR_pipeline_A_hc50.csv (output of script 3, mRMR)

Output:
    - lasso_pipeline_A_HC50_features.csv - surviving features + meta columns,
      all rows (train + test)
    - lasso_pipeline_A_HC50_coefficients.csv - every feature, its LASSO
      coefficient, and whether it was kept
    - lasso_pipeline_A_HC50_coef_plot.png - bar chart, all surviving features
    - lasso_path_plot_pipeline_A_hc50 - regularization path, all features
    - lasso_path_top10_pipeline_A_hc50 / lasso_path_bottom10_pipeline_A_hc50 -
      regularization path, top/bottom 10 by |coefficient|
    - lasso_bar_top10_pipeline_A_hc50 / lasso_bar_bottom10_pipeline_A_hc50 -
      bar charts, top/bottom 10 by |coefficient|
    - lasso_correlation_heatmap_pipeline_A_hc50 - correlation heatmap,
      surviving features
    - lasso_correlation_heatmap_excluded_pipeline_A_hc50 - correlation
      heatmap, excluded features
    - Printed: feature matrix shape, train rows/feature count, LASSO runtime,
      counts of features before/after/excluded

"""

# LASSO
# input: mrmr_selected_features (output of script 3 mutual info || F-stat)
# output: lasso_final_features.csv and lasso_coefficirnts.csv

import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from time import time

# paths
BASE_DIR = "/mnt/c/Users/najla/Downloads/SCOL 391 - Mansbach/CleaningDBsScripts/dbaasp_grampa_hemolytik_hc50_mic_ANALYSIS"
#BASE_DIR   = "/mnt/c/Users/najla/Downloads/SCOL 391 - Mansbach/CleaningDBsScripts/dbaasp_grampa_hemolytik_hc50_mic_ANALYSIS/FSTAT" # for now, let's focus on mutual info
#INPUT_CSV  = os.path.join(BASE_DIR, "features_mrmrMRMR.csv") # either from mutual info or from fstat
INPUT_CSV  = os.path.join(BASE_DIR, "features_mRMR_pipeline_A_hc50.csv")
OUT_FEAT   = os.path.join(BASE_DIR, "lasso_pipeline_A_HC50_features.csv")
OUT_COEF   = os.path.join(BASE_DIR, "lasso_pipeline_A_HC50_coefficients.csv")
OUT_PLOT   = os.path.join(BASE_DIR, "lasso_pipeline_A_HC50_coef_plot.png")

# load the data
print("Debugging: loading mRMR selected features")
df = pd.read_csv(INPUT_CSV)
# commenting these three lines out because in the new mRMR files, the hc50 and mic values are included.
#df_raw = pd.read_excel(EXCEL_PATH, engine="openpyxl")
#df = df.merge(df_raw[["sequence", "log10hc50"]], left_on="Sequence", right_on="sequence", how="left")
#df = df.drop(columns=["sequence"])


LABEL_COL = "Class" # this is the target column ("selective" "inactive" etc.)
ID_COL = ["SequenceIndex", "Sequence"] # this is for the columns that are non-features
TARGET = "log10hc50"
META = [LABEL_COL, "log10hc50", "log10mic", "Split"] + [c for c in ID_COL if c in df.columns]
drop_cols = [c for c in META if c in df.columns]

X = df.drop(columns=drop_cols)
y = df[TARGET]

# LabelEncode maps the 4 classes into integers as LassoCV needs numeric digits to operate. Output is for debugging.
#le = LabelEncoder()
#y = le.fit_transform(y_raw)
y = df[TARGET]
#print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"Feature matrix shape: {X.shape}")

# Application of LASSO, the model adds a penalty to the loss function proportional to the sum of absolute coefficients values
# The penalty L1 forces less informative features to have their coefficients shrunk all the way to 0.
# CV basically tries a range of regularization strengths (lambda/alpha) and picks the one with the best performance.

train_mask = df["Split"] == "train"
X_train = X.loc[train_mask]
y_train = y.loc[train_mask]
print(f"Fitting LASSO on {len(X_train)} train rows ({X_train.shape[1]} features).")

t0 = time()
print("Starting LASSO...")
lasso = LassoCV(
    cv=5,
    max_iter=10_000, #Scikit learn default is 1000, but I decided to go with 10000 because there is simply too many features and it will cause a Convergence warning
    n_jobs=-1,
    random_state=42
)
lasso.fit(X_train, y_train)
t1 = time()
time_taken = (t1-t0)
print(f"End of Lasso. {time_taken}")
print(f"LASSO took: {time_taken}")
coefficient_series = pd.Series(lasso.coef_, index=X.columns)
surviving_features = coefficient_series[coefficient_series !=0]
excluded_features_after_LASSO = coefficient_series[coefficient_series==0] # Since LASSO shrunk the less informative features, coefficients are at 0.
print(f"\nFeatures before LASSO: {X.shape[1]}")
print(f"Features after  LASSO: {len(surviving_features)}")
print(f"Features excluded after LASSO: {len(excluded_features_after_LASSO)}")
df_surviving_features = df[drop_cols + list(surviving_features.index)]
df_surviving_features.to_csv(OUT_FEAT, index=False) # saving the final feature matrix
coef_df = coefficient_series.reset_index()
coef_df.columns = ["feature", "lasso_coefficient"]
coef_df["kept"] = coef_df["lasso_coefficient"] != 0
coef_df = coef_df.sort_values("lasso_coefficient", key=abs, ascending=False)
coef_df.to_csv(OUT_COEF, index=False) # False means it was excluded, true means LASSO kept the feature.

# coefficient path plot
from sklearn.linear_model import lasso_path
alphas_path, coefs_path, _ = lasso_path(X_train, y_train, alphas=lasso.alphas_, max_iter=10_000)
fig, ax = plt.subplots(figsize=(12,6))
for coef_row in coefs_path:
    ax.plot(alphas_path, coef_row, linewidth=0.7, alpha=0.6)
ax.axvline(lasso.alpha_, color="black", linestyle="--", linewidth=1.5,
           label=f"Selected α = {lasso.alpha_:.5f}")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("α (log scale, decreasing -->)")
ax.set_ylabel("Coefficient value")
ax.set_title("LASSO Regularization Path (post-mRMR features) (HC50)")
ax.legend()
plt.tight_layout()

OUT_PATH_PLOT = os.path.join(BASE_DIR, "lasso_path_plot_pipeline_A_hc50")
plt.savefig(OUT_PATH_PLOT, dpi=150)
plt.close()
print(f"Saved path plot --> {OUT_PATH_PLOT}")
# 2 bar chart
fig, ax = plt.subplots(figsize=(10, max(4, len(surviving_features) * 0.3)))
colors = ["steelblue" if v > 0 else "tomato" for v in surviving_features.values]
ax.barh(surviving_features.index, surviving_features.values, color=colors)
ax.set_xlabel("LASSO Coefficient")
ax.set_title(f"Surviving Features after mRMR + LASSO  (α = {lasso.alpha_:.5f}) (HC50)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()

plt.savefig(OUT_PLOT, dpi=150)
plt.close()
print(f"Saved bar chart --> {OUT_PLOT}")
# 1a top 10 coefficient path plot
from sklearn.linear_model import lasso_path
alphas_path, coefs_path, _ = lasso_path(X_train, y_train, alphas=lasso.alphas_, max_iter=10_000)
coef_at_chosen = pd.Series(lasso.coef_, index=X.columns)
sorted_by_abs = coef_at_chosen.abs().sort_values(ascending=False)
top10_names    = sorted_by_abs.head(10).index.tolist()
bottom10_names = sorted_by_abs.tail(10).index.tolist()
top10_idx    = [X.columns.get_loc(n) for n in top10_names]
bottom10_idx = [X.columns.get_loc(n) for n in bottom10_names]

# top10 coefficient path
fig, ax = plt.subplots(figsize=(12, 6))
for i, name in zip(top10_idx, top10_names):
    ax.plot(alphas_path, coefs_path[i], linewidth=1.5, label=name)
ax.axvline(lasso.alpha_, color="black", linestyle="--", linewidth=1.5,
           label=f"Selected α = {lasso.alpha_:.5f}")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("α (log scale, decreasing -->)")
ax.set_ylabel("Coefficient value")
ax.set_title("LASSO Path — Top 10 Features by |Coefficient| (HC50)")
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
OUT_PATH_TOP10 = os.path.join(BASE_DIR, "lasso_path_top10_pipeline_A_hc50")
plt.savefig(OUT_PATH_TOP10, dpi=150)
plt.close()
print(f"Saved top 10 path plot --> {OUT_PATH_TOP10}")

# 1b bottom 10 coefficient path
fig, ax = plt.subplots(figsize=(12, 6))
for i, name in zip(bottom10_idx, bottom10_names):
    ax.plot(alphas_path, coefs_path[i], linewidth=1.5, label=name)
ax.axvline(lasso.alpha_, color="black", linestyle="--", linewidth=1.5,
           label=f"Selected α = {lasso.alpha_:.5f}")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("α (log scale, decreasing -->)")
ax.set_ylabel("Coefficient value")
ax.set_title("LASSO Path — Bottom 10 Features by |Coefficient| (HC50)")
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
OUT_PATH_BOT10 = os.path.join(BASE_DIR, "lasso_path_bottom10_pipeline_A_hc50")
plt.savefig(OUT_PATH_BOT10, dpi=150)
plt.close()
print(f"Saved bottom 10 path plot --> {OUT_PATH_BOT10}")

# 2a top 10
top10_surviving = surviving_features.reindex(
    surviving_features.abs().sort_values(ascending=False).head(10).index
)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["steelblue" if v > 0 else "tomato" for v in top10_surviving.values]
ax.barh(top10_surviving.index, top10_surviving.values, color=colors)
ax.set_xlabel("LASSO Coefficient")
ax.set_title(f"Top 10 Surviving Features by |Coefficient|  (α = {lasso.alpha_:.5f}) (HC50)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
OUT_BAR_TOP10 = os.path.join(BASE_DIR, "lasso_bar_top10_pipeline_A_hc50")
plt.savefig(OUT_BAR_TOP10, dpi=150)
plt.close()
print(f"Saved top 10 bar chart --> {OUT_BAR_TOP10}")
# plot 2b bottom 10
bottom10_surviving = surviving_features.reindex(
    surviving_features.abs().sort_values(ascending=True).head(10).index
)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["steelblue" if v > 0 else "tomato" for v in bottom10_surviving.values]
ax.barh(bottom10_surviving.index, bottom10_surviving.values, color=colors)
ax.set_xlabel("LASSO Coefficient")
ax.set_title(f"Bottom 10 Surviving Features by |Coefficient|  (α = {lasso.alpha_:.5f}) (HC50)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
OUT_BAR_BOT10 = os.path.join(BASE_DIR, "lasso_bar_bottom10_pipeline_A_hc50")
plt.savefig(OUT_BAR_BOT10, dpi=150)
plt.close()
print(f"Saved bottom 10 bar chart --> {OUT_BAR_BOT10}")

# plot 2c bar chart all of them
fig, ax = plt.subplots(figsize=(10, max(4, len(surviving_features) * 0.3)))
colors = ["steelblue" if v > 0 else "tomato" for v in surviving_features.values]
ax.barh(surviving_features.index, surviving_features.values, color=colors)
ax.set_xlabel("LASSO Coefficient")
ax.set_title(f"All Surviving Features Post mRMR + LASSO  (α = {lasso.alpha_:.5f}) (HC50)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150)
plt.close()
print(f"Saved full bar chart --> {OUT_PLOT}")
# plot 3 correlation heatmap of surviving features
import seaborn as sns
surviving_df = df[list(surviving_features.index)]
corr_matrix = surviving_df.corr()

fig, ax = plt.subplots(figsize=(max(8, len(surviving_features) * 0.5),
                                max(6, len(surviving_features) * 0.5)))
sns.heatmap(
    corr_matrix,
    ax=ax,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.3,
    annot=len(surviving_features) <= 20,
    fmt=".2f"
)
ax.set_title("Correlation Matrix - Surviving Features Post LASSO (HC50)")
plt.tight_layout()
OUT_HEATMAP = os.path.join(BASE_DIR, "lasso_correlation_heatmap_pipeline_A_hc50")
plt.savefig(OUT_HEATMAP, dpi=150)
plt.close()
print(f"Saved heatmap --> {OUT_HEATMAP}")
# plot 4 correlation heatmap of excluded features
excluded_df = df[list(excluded_features_after_LASSO.index)]
corr_matrix_excl = excluded_df.corr()

fig, ax = plt.subplots(figsize=(max(12, len(excluded_features_after_LASSO) * 0.15),
                                max(10, len(excluded_features_after_LASSO) * 0.15)))
sns.heatmap(
    corr_matrix_excl,
    ax=ax,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0,
    annot=False,
    xticklabels=False,
    yticklabels=False
)
ax.set_title("Correlation Matrix — 194 Excluded Features Post LASSO (HC50)")
plt.tight_layout()
OUT_HEATMAP_EXCL = os.path.join(BASE_DIR, "lasso_correlation_heatmap_excluded_pipeline_A_hc50")
plt.savefig(OUT_HEATMAP_EXCL, dpi=150)
plt.close()
print(f"Saved excluded heatmap --> {OUT_HEATMAP_EXCL}")
