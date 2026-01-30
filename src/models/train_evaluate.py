import joblib
import os
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.config.config import (
    RANDOM_STATE,
    MODELS_DIR,
    N_SPLITS,
    N_ITER,
    SCORING,
)


def train_and_evaluate(model_name, model, param_grid, X_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    search.fit(X_train, y_train)

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(search.best_estimator_, model_path)

    return search
