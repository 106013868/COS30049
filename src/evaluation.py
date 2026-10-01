import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.base import clone
import make_fake_domains as mfd
import config as c

def grouped_split(df, seed=42):
    targets = df[c.LABEL_COLUMN]
    groups = df[c.DOMAIN_COLUMN]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, targets, groups))
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    return train, test

def random_split(df, seed=42):
    train, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[c.LABEL_COLUMN])
    
    return train, test

def evaluate_model(model, train, test):
    model_copy = clone(model)
    train_features = train[c.FEATURE_COLUMNS]
    train_labels = train[c.LABEL_COLUMN]
    test_features = test[c.FEATURE_COLUMNS]
    test_labels = test[c.LABEL_COLUMN]

    model_copy.fit(train_features, train_labels)
    pred = model_copy.predict(test_features)

    return score_predictions(test_labels, pred)

def score_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    t_neg, f_pos, f_neg, t_pos = confusion_matrix(y_true, y_pred).ravel()

    return {
        "acc": accuracy, "prec": precision, "recall": recall, "f1": f1, "t_neg": t_neg, "f_pos": f_pos, "f_neg": f_neg, "t_pos": t_pos}
    
if __name__ == "__main__":
    print("Not to be run directly")
    from models import get_models
    data = mfd.make_dataset()
    train, test = grouped_split(data)
    logReg = get_models()["LogisticRegression"]

    print(evaluate_model(logReg, train, test))