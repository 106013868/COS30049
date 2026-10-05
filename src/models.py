"""
models.py
defines candidate models for comparison
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_models(seed=42):
    """
    returns dict of model name to untrained model
    """
    dummy = DummyClassifier(strategy="most_frequent")
    logReg = LogisticRegression(random_state=seed, max_iter=1000)
    logRegwithscaler = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000))
    balancedLogReg = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced"))
    rf = RandomForestClassifier(random_state=seed, n_estimators=100, max_depth=20, n_jobs=-1, class_weight="balanced")
    hgb = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    hgb_unbalanced = HistGradientBoostingClassifier(random_state=seed)

    return {
        "dummy": dummy,
        "LogisticRegression": logReg,
        "LogisticRegressionWithScaler": logRegwithscaler,
        "BalancedLogisticRegression": balancedLogReg,
        "RandomForest": rf,
        "HistGradientBoosting": hgb,
        "HGB_Unbalanced": hgb_unbalanced
    }

if __name__ == "__main__":
    print(get_models())