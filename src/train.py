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

def load_model(path=cfg.MODEL_PATH):
    """
    loads saved model from path
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

def predict(model, df):
    """
    adds prediction, probability and trust score columns to a copy of df
    """
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
    """
    In: raw url
    Out: one row dataframe with prediction and trust score
    """
    cleaned = clean_url(url)
    has_https = check_https(url)
    features = extract_features(cleaned, has_https)
    row = pd.DataFrame([features])
    row[cfg.URL_COLUMN] = cleaned

    return predict(model, row)

if __name__ == "__main__":
    data = load_data()
    model = get_models()["HistGradientBoosting"]
    trained = train_final_model(model, data)
    save_model(trained)

    loaded_model = load_model()
    test_urls = [    "google.com",
    "https://www.google.com/search?q=weather+melbourne",
    "github.com",
    "https://github.com/scikit-learn/scikit-learn",
    "en.wikipedia.org/wiki/Phishing",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "amazon.com.au",
    "https://www.amazon.com/gp/help/customer/display.html",
    "abc.net.au/news",
    "https://www.bom.gov.au/vic/forecasts/melbourne.shtml",
    "swinburne.edu.au",
    "https://www.swin.edu.au/study/find-a-course",
    "ato.gov.au",
    "https://www.commbank.com.au/banking/netbank.html",
    "ptv.vic.gov.au/journey",
    "docs.python.org/3/library/pathlib.html",
    "https://stackoverflow.com/questions/tagged/pandas",
    "microsoft.com",
    "https://support.apple.com/en-au/iphone",
    "reddit.com/r/melbourne"]
    for url in test_urls:
        result = predict_url(loaded_model, url)
        print(f"URL: {url}, Prediction: {result[cfg.PREDICTION_COLUMN].values[0]}, Trust Score: {result[cfg.TRUST_COLUMN].values[0]}%")