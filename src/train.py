from sklearn.base import clone
import config as cfg
from prepare_data import load_data
from models import get_models
import joblib
import pandas as pd
from data_processing import extract_features

def train_final_model(model, df):
    final_model = clone(model)
    features = df[cfg.FEATURE_COLUMNS]
    labels = df[cfg.LABEL_COLUMN]
    final_model.fit(features, labels)

    return final_model

def save_model(model, path=cfg.MODEL_PATH):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path=cfg.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

def predict(model, df):
    features = df[cfg.FEATURE_COLUMNS]
    probabilities = model.predict_proba(features)[:, 1]
    labels = model.predict(features)
    trust = ((1 - probabilities) * 100).round()
    results = df.copy()
    results[cfg.PREDICTION_COLUMN] = labels
    results[cfg.PROBABILITY_COLUMN] = probabilities
    results[cfg.TRUST_COLUMN] = trust
    
    return results

def predict_url(model, url):
    # strips leading https/http and trailing slashes, www. converts to lowercase, and removes whitespace
    cleaned = str(url).strip().lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")
    features = extract_features(cleaned)
    row = pd.DataFrame([features])
    row[cfg.URL_COLUMN] = cleaned

    return predict(model, row)

if __name__ == "__main__":
    data = load_data()
    model = get_models()["HistGradientBoosting"]
    trained = train_final_model(model, data)
    save_model(trained)

    loaded_model = load_model()
    test_urls = ["http://www.google.com", "google.com/search?q=weather", "google.com", "www.google.com/", "http://google.com"]
    for url in test_urls:
        result = predict_url(loaded_model, url)
        print(f"URL: {url}, Prediction: {result[cfg.PREDICTION_COLUMN].values[0]}, Trust Score: {result[cfg.TRUST_COLUMN].values[0]}%")