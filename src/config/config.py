import os

RANDOM_STATE = 42

DATA_PATH = "data/raw/creditcard.csv"

RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

N_SPLITS = 3
N_ITER = 6
SCORING = "roc_auc"

for d in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)
