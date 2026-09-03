# HealthConnect - Week 5 Project Summary (Data Science)

## Completed work

Built and evaluated a binary no-show baseline using **4,737** non-cancelled appointments. The original files were not modified. The target is `No-Show = 1` and `Attended = 0`; cancellations remain excluded as agreed in Week 4.

## Data preparation and feature engineering

- `reminder_channel` missingness is retained as the meaningful category `None`.
- Created `prior_no_show_rate`, `has_prior_no_show`, `lead_time_bucket`, `booking_month` and `appointment_month`.
- Excluded IDs, outcome fields, raw dates and `waiting_time_minutes`. The latter remains a leakage risk because it may not be known at booking time.
- Used a grouped train/test split by `patient_id`, with **zero** shared patients between partitions.

## Initial evaluation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.621 | 0.620 | 0.651 | 0.635 | 0.676 |

## Findings and recommendations

- No-show rate rises from **29.5%** for 0-7 days to **63.9%** for 31+ days lead time.
- Patients with a prior no-show have a rate of **57.8%**, compared with **46.3%** otherwise.
- Use any risk score for proportionate support (for example, an additional reminder or administrative call), never to deny care or penalise a patient.
- Data Analytics can monitor lead-time and prior-attendance segments as dashboard KPIs; Data Science provides the target and model metrics.

## Limitations and Week 6

This is synthetic data, one grouped hold-out split and a provisional 0.50 threshold. Week 6 should add grouped cross-validation, calibration, threshold selection based on intervention capacity, a tree-model comparison and fairness review.
