from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False


def get_models(random_state=42):
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    return models
