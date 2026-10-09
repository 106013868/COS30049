"""
train.py
trains, saves and loads final model, predicts on dataframes or single urls
"""
from sklearn.base import clone
import config as cfg
from prepare_data import load_data
from models import get_models
import joblib
import pandas as pd
from data_processing import extract_features, clean_url, check_https

def train_final_model(model, df):
    """
    trains a copy of the model on the full dataset
    """
    final_model = clone(model)
    features = df[cfg.FEATURE_COLUMNS]
    labels = df[cfg.LABEL_COLUMN]
    final_model.fit(features, labels)

    return final_model

def save_model(model, path=cfg.MODEL_PATH):
    """
    saves model to path, creates folder if missing
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

if __name__ == "__main__":
    # train final model on the full dataset and save it
    data = load_data()
    model = get_models()[cfg.FINAL_MODEL]
    trained = train_final_model(model, data)
    save_model(trained)