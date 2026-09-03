[README.md](https://github.com/user-attachments/files/31802887/README.md)
# HealthConnect Experience Lab - Week 5 Data Science

## Deliverables

- `Week5_Data_Science_Baseline_Modelling.ipynb` - baseline modelling notebook
- `Week5_Project_Summary.md` - required concise summary
- `data/` - a separate copy of the provided model-ready source and derived features
- `figures/` - five decision-supporting visualisations
- `evaluation_metrics.json` - reproducible initial metrics
- `src/week5_analysis.py` - dependency-free reproducible analysis implementation

## Model

Logistic Regression predicts no-show risk using booking-time and prior-attendance information. The test split has no patient overlap with the training split. Initial ROC-AUC: **0.676**. This is a synthetic-data baseline, not a deployable clinical model.
