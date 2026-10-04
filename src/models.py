from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_models(seed=42):
    dummy = DummyClassifier(strategy="most_frequent")
    logReg = LogisticRegression(random_state=seed, max_iter=1000)
    logRegwithscaler = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000))
    balancedLogReg = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000, class_weight="balanced"))
    
    return {"dummy": dummy, "LogisticRegression": logReg, "LogisticRegressionWithScaler": logRegwithscaler, "BalancedLogisticRegression": balancedLogReg}

if __name__ == "__main__":
    print(get_models())