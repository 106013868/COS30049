import argparse
import joblib
import pandas as pd
import config as cfg
from data_processing import clean_url, check_https, extract_features

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
    # probability of class 1 (malicious)
    probabilities = model.predict_proba(features)[:, 1]
    labels = model.predict(features)
    # trust score is the chance the url is not malicious, as a percentage
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
    # same cleaning steps as data_processing
    cleaned = clean_url(url)
    # https is read from the raw url because clean_url removes the protocol
    has_https = check_https(url)
    features = extract_features(cleaned, has_https)
    row = pd.DataFrame([features])
    row[cfg.URL_COLUMN] = cleaned

    return predict(model, row)

def parse_args():
    """
    parse url args passed into the script to be used for predictions
    """
    parser = argparse.ArgumentParser(description="This module accepts URLs as arguments when executing the script and outputs predictions on whether the URL is safe or malicious.")
    parser.add_argument("urls", nargs="+", help='Enter URL(s) inside double quotation marks separated by spaces. Example: "https://google.com"')

    return parser.parse_args()

if __name__ == "__main__":
    # pull in url arguments from user
    args = parse_args()
    # load the saved model
    model = load_model()

    for url in args.urls:
        result = predict_url(model, url)
        label = "phishing" if result[cfg.PREDICTION_COLUMN].values[0] == 1 else "safe"
        trust = int(result[cfg.TRUST_COLUMN].values[0])

        print(f"URL: {url}, Prediction: {label}, Trust Score: {trust}%")