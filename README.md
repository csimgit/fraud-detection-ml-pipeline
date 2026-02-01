#### Fraud Detection ML Pipeline

End-to-end machine learning pipeline for credit card fraud detection using the
Kaggle Credit Card Fraud dataset.

This project demonstrates a Enterprise-level ML system, not just notebooks.



#### Problem
Credit card fraud detection is a highly imbalanced classification problem
where fraudulent transactions represent less than 0.2% of total volume.

Traditional accuracy-based evaluation fails in such scenarios.

---

#### Objectives
- Handle extreme class imbalance
- Build scalable ML pipelines
- Compare multiple algorithms
- Optimize fraud recall and precision
- Demonstrate enterprise ML engineering practices



#### Techniques Used
- Stratified train/test splitting
- Feature scaling
- SMOTE oversampling
- Multiple ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost (optional)
- Hyperparameter tuning (RandomizedSearchCV)
- ROC-AUC and PR-AUC evaluation
- Threshold optimization
- Feature importance analysis

---

#### Create Environment
.env
venv/

#### How to Run
```bash
pip install -r requirements.txt
python run_pipeline.py

