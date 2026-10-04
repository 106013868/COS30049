from pathlib import Path

# Constant definitions
DATA_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = DATA_DIR / "models"
DATASET_PATH = DATA_DIR / "datasets" / "processed_dataset.csv"
FEATURES_PATH = DATA_DIR / "datasets" / "processed_features.csv"
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"
URL_COLUMN = "url"
LABEL_COLUMN = "label"
DOMAIN_COLUMN = "domain"
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
CLUSTER_COLUMN = "cluster"
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "probability"
TRUST_COLUMN = "trust_score"