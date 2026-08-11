# peptide-drug-classifier

Classifying antimicrobial peptides into four activity/toxicity classes (Selective, Pure Hemolytic, Both Active/Toxic, Inactive) based on E. coli MIC and erythrocyte HC50 values, using physicochemical descriptors + SVM. Undergraduate research project (SCOL 391) in Dr. Mansbach's lab, Concordia University.

## What's in here

- **`data/`** : the merged DBAASP + GRAMPA + Hemolytik dataset (4810 peptides, MIC + HC50 labeled).
- **`pipelineA/`** : leakage-safe pipeline. Train/test split happens first, then scaling, mRMR, and LASSO are fit on the training partition only.
- **`pipelineB/`** : leakage-prone pipeline, kept as a diagnostic. Scaling/feature selection are fit on the full dataset before splitting, so we can measure how much this inflates apparent performance.
- **`shared/`** : hyperparameter search scripts and the peptide analyzer (class distribution plots etc.), used by both pipelines.
- **`mutualinfo/`** : early mRMR/LASSO exploration runs, kept for reference.
- **`results/`** : final outputs (plots, LASSO coefficients, SHAP values, model metadata) for both pipelines, split by target (HC50, MIC, union).
- **`old_pipelines/`** : first-draft feature selection scripts from before Pipeline A/B existed. Not maintained, kept for history.

## Pipeline steps (both A and B)

1. `step0_split.py` : train/test split (80/20, stratified, seed=42)
2. `scaling.py` : StandardScaler
3. `mRMR.py` : mRMR feature selection (capped at k=300)
4. `lasso_hc50.py` / `lasso_mic.py` : LASSO regression, separate for each target
5. `SVM_hc50.py` / `SVM_mic.py` (+ `_with_SHAP` versions) : SVM classifier per target, with SHAP explanation
6. `union.py` : combined classifier using both HC50 and MIC selected features
7. `SMOTE*.py` (Pipeline A only) : oversampling diagnostic for the minority class

Full methods are described in the report.

## Notes

- Feature extraction uses ProPy3 (1547 descriptors → 1528 after dropping zero-variance features).
- The two pipelines exist to test whether fitting feature selection before vs. after the train/test split changes results : turns out accuracy alone doesn't reliably show leakage, but SHAP feature overlap between classes does.
- Large intermediate files (full mRMR/LASSO feature matrices, tens of MB each) aren't tracked in this repo : they're regenerable by rerunning the pipeline scripts on the data in `data/`.
