# Customer Churn Analysis & Prediction

Starter project for exploring customer churn and building prediction models.

Structure
- `data/` — raw and processed datasets (not in repo)
- `notebooks/` — exploratory notebooks
- `src/churn_prediction/` — package source code
- `reports/` — generated reports and figures

Quick start

1. Create a virtual environment (recommended): `python -m venv .venv`
2. Activate it (Windows): `.venv\Scripts\activate` or (PowerShell): `. .venv\Scripts\Activate.ps1`
3. Install deps:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

4. Run training script or open notebooks.

Templates
-
	- EDA notebook template: `notebooks/eda_template.ipynb` — quick cells to load data, show head, missing values, and churn distribution.
	- Run script: `scripts/run_training.ps1` — activates `.venv`, installs `requirements.txt`, and runs the training module.

Production model
-
	- **Chosen model:** `reports/model_class_weight.joblib` (persisted as `reports/model_production.joblib`).
	- **Why chosen:** `class_weight='balanced'` gave the best minority-class F1 in our comparison.
	- **Key test metrics:** accuracy 0.7658, precision 0.5495, recall 0.6524, F1 0.5966, ROC AUC ~0.8275.
	- **Artifacts:** metrics are in `reports/imbalance_results.json`; evaluated model artifacts are in `reports/`.
