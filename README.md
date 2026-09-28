# Diabetes Prediction with the Pima Indians Dataset

A machine learning project that predicts the onset of diabetes from diagnostic measurements, and tests whether a complex model (gradient boosting) beats a simple one (logistic regression).

## Dataset
Pima Indians Diabetes Database from Kaggle: 768 patients, 8 measurements (glucose, BMI, age, blood pressure, insulin, and others) and a diabetes outcome. About 35% of patients are diabetic.

## Method
- Zeros in glucose, blood pressure, skin thickness, insulin and BMI are impossible values, so they were treated as missing.
- 80/20 train/test split, stratified by outcome.
- Pipeline: imputation, random over-sampling of the minority class, then the classifier. Over-sampling is applied only to training folds.
- Both models tuned with GridSearchCV (5-fold).
- Libraries: pandas, numpy, scikit-learn, imbalanced-learn.

## Results

| Metric (test set) | Logistic Regression | Gradient Boosting |
|---|---|---|
| Accuracy | 0.734 | 0.766 |
| Precision (diabetic) | 0.603 | 0.636 |
| Recall (diabetic) | 0.704 | 0.778 |
| F1 (diabetic) | 0.650 | 0.700 |

Cross-validated accuracy was identical for both models (0.761). Repeated cross-validation gave mean recall of 0.713 ± 0.07 for logistic regression and 0.702 ± 0.055 for gradient boosting.

## Conclusion
The two models performed equivalently. Gradient boosting's higher test-set scores are best explained by the small test set (154 rows). Logistic regression is preferable because it is simpler and easier to interpret clinically.

## Limitations
Small sample, heavily missing insulin values, and a single train/test split. This is a learning project on a public dataset, not a clinical tool.

## How to run
Open the notebook in Kaggle or Jupyter, attach the Pima Indians Diabetes dataset, and run all cells.
