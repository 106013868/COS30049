from sklearn.base import clone
import config as cfg
from prepare_data import load_data
from models import get_models
import joblib

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


if __name__ == "__main__":
    data = load_data()
    model = get_models()["BalancedLogisticRegression"]
    trained = train_final_model(model, data)

    loaded_model = load_model()
    preview = predict(loaded_model, data.head(10))

    print(preview[[cfg.URL_COLUMN, cfg.LABEL_COLUMN, cfg.PREDICTION_COLUMN, cfg.PROBABILITY_COLUMN, cfg.TRUST_COLUMN]].to_string())