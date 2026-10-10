"""
predict.py
command-line tool to score URLs with the trained PhishLens model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config as cfg
from inference import load_model, predict_url

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