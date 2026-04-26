"""
This script implements LASSO regression from Scikit-Learn and it follows previous inputs from feature selection and mRMR.
This outputs 2 CSV files:
    1) All surviving features.
    2) All features with a boolean column, which shows what was excluded and included in the first CSV.
    3) A bar chart.

Run it using this command:
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
BASE_DIR   = "/mnt/c/Users/najla/Downloads/SCOL 391 - Mansbach/CleaningDBsScripts/dbaasp_grampa_hemolytik_hc50_mic_ANALYSIS/MUTUALINFO"
#BASE_DIR   = "/mnt/c/Users/najla/Downloads/SCOL 391 - Mansbach/CleaningDBsScripts/dbaasp_grampa_hemolytik_hc50_mic_ANALYSIS/FSTAT" # for now, let's focus on mutual info
INPUT_CSV  = os.path.join(BASE_DIR, "features_mrmrMRMR.csv") # either from mutual info or from fstat
OUT_FEAT   = os.path.join(BASE_DIR, "lasso_features.csv")
OUT_COEF   = os.path.join(BASE_DIR, "lasso_coefficients.csv")
OUT_PLOT   = os.path.join(BASE_DIR, "lasso_coef_plot.png")

# load the data
print("Debugging: loading mRMR selected features")
df = pd.read_csv(INPUT_CSV)
LABEL_COL = "Class" # this is the target column ("selective" "inactive" etc.)
ID_COL = ["SequenceIndex", "Sequence"] # this is for the columns that are non-features
drop_cols  = [LABEL_COL] + [c for c in ID_COL if c in df.columns]
X          = df.drop(columns=drop_cols)
y_raw      = df[LABEL_COL]

# LabelEncode maps the 4 classes into integers as LassoCV needs numeric digits to operate. Output is for debugging.
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"Feature matrix shape: {X.shape}")

# Application of LASSO, the model adds a penalty to the loss function proportional to the sum of absolute coefficients values
# The penalty L1 forces less informative features to have their coefficients shrunk all the way to 0.
# CV basically tries a range of regularization strengths (lambda/alpha) and picks the one with the best performance.
t0 = time()
print("Starting Lasso...")
lasso = LassoCV(
    cv=5,
    max_iter = 10_000, #Scikit learn default is 1000, but I decided to go with 10000 because there is simply too many features and it will cause a Convergence warning
    n_jobs=-1,
    random_state=42
)
lasso.fit(X,y)
t1 = time()
time_taken = (t1-t0)
print("End of Lasso.")
print(f"LASSO took: {time_taken}")
coefficient_series = pd.Series(lasso.coef_, index=X.columns)
surviving_features = coefficient_series[coefficient_series !=0]
excluded_features_after_LASSO = coefficient_series[coefficient_series==0] # Since LASSO shrunk the less informative features, their coefficients are at 0.
print(f"\nFeatures before LASSO: {X.shape[1]}")
print(f"Features after  LASSO: {len(surviving_features)}")
print(f"Features excluded after LASSO: {len(excluded_features_after_LASSO)}")
df_surviving_features = df[list(surviving_features.index) + [LABEL_COL]]
df_surviving_features.to_csv(OUT_FEAT, index=False) # saving the final feature matrix
coef_df = coefficient_series.reset_index()
coef_df.columns = ["feature", "lasso_coefficient"]
coef_df["kept"] = coef_df["lasso_coefficient"] != 0
coef_df = coef_df.sort_values("lasso_coefficient", key=abs, ascending=False)
coef_df.to_csv(OUT_COEF, index=False) # False means it was excluded, true means LASSO kept the feature.

# Making the plot
fig, ax = plt.subplots(figsize=(10, max(4, len(surviving_features) * 0.3)))
colors = ["blue" if v > 0 else "red" for v in surviving_features.values]
ax.barh(surviving_features.index, surviving_features.values, color=colors) # features name on the y axis, coefficient values on the x axis, and colors above of it.
ax.set_xlabel("LASSO Coefficient") # Label for X-Axis
ax.set_title(f"Surviving Features Post mRMR and LASSO Regression (λ = {lasso.alpha_:.5f})") # Title
ax.axvline(0, color="black", linewidth=0.8) # add a line in the center
plt.tight_layout() # to make it pretty
plt.savefig(OUT_PLOT)
