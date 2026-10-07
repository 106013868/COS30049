"""
config.py
holds constants used throughout all modules
i.e. paths, data column names
"""
from pathlib import Path

# Constant definitions
ROOT_DIR = Path(__file__).resolve().parent.parent

# folder paths
MODELS_DIR = ROOT_DIR / "models"
EVAL_DIR = ROOT_DIR / "evaluation"
TABLES_DIR = EVAL_DIR / "tables"
PLOTS_DIR = EVAL_DIR / "plots"

# processed data and saved model paths
DATASET_PATH = ROOT_DIR / "datasets" / "processed_dataset.csv"
FEATURES_PATH = ROOT_DIR / "datasets" / "processed_features.csv"
MODEL_PATH = MODELS_DIR / "hist_gradient_boosting.joblib"
FINAL_MODEL = "HistGradientBoosting" # chosen model after evaluation

# dataset column names
URL_COLUMN = "url"
LABEL_COLUMN = "label"
DOMAIN_COLUMN = "domain"
# features the model trains on
FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "path_length",
    "digit_count",
    "special_character_count",
    "dot_count",
    "subdomain_count",
    "has_ip",
    "suspicious_keyword_count"
]
# column names added during clustering and prediction
CLUSTER_COLUMN = "cluster"
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "probability"
TRUST_COLUMN = "trust_score"