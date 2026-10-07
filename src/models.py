"""
models.py
defines candidate models for comparison
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def get_models(seed=42):
    """
    returns dict of model name to untrained model
    """
    # baseline
    logReg = LogisticRegression(random_state=seed, max_iter=1000)
    # baseline with scaling
    logRegwithscaler = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000))
    # scaling and balanced class weights
    balancedLogReg = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced"))
    # 100 trees, depth capped at 20, balanced class weights
    rf = RandomForestClassifier(random_state=seed, n_estimators=100, max_depth=20, n_jobs=-1, class_weight="balanced")
    # final model, balanced class weights
    hgb = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    # same as hgb without class weights
    hgb_unbalanced = HistGradientBoostingClassifier(random_state=seed)
    # two hidden layers (64, 32), stops early when the validation score stops improving
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=seed))

    return {
        "LogisticRegression": logReg,
        "LogisticRegressionWithScaler": logRegwithscaler,
        "BalancedLogisticRegression": balancedLogReg,
        "RandomForest": rf,
        "HistGradientBoosting": hgb,
        "HGB_Unbalanced": hgb_unbalanced,
        "NeuralNetwork": mlp
    }

if __name__ == "__main__":
    # prints the model definitions
    print(get_models())