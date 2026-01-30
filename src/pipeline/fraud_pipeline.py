from sklearn.model_selection import train_test_split

from src.data.data_loader import load_data
from src.preprocessing.preprocessing import preprocess
from src.models.model_registry import get_models
from src.models.train_evaluate import train_and_evaluate
from src.evaluation.metrics import evaluate
from src.config.config import RANDOM_STATE


def run_pipeline():
    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = get_models()

    param_grids = {
        "logistic": {"model__C": [0.01, 0.1, 1, 10]},
        "decision_tree": {"model__max_depth": [5, 10, None]},
        "random_forest": {"model__max_depth": [10, 20, None]},
        "gradient_boosting": {"model__learning_rate": [0.01, 0.1]},
        "xgboost": {"model__max_depth": [3, 5, 8]},
    }

    results = {}

    for name, model in models.items():
        grid = param_grids.get(name, {})
        search = train_and_evaluate(name, model, grid, X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        results[name] = metrics["roc_auc"]

    print("Final ROC-AUC:", results)
