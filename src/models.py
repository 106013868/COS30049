from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

def get_models(seed=42):
    dummy = DummyClassifier(strategy="most_frequent")
    logReg = LogisticRegression(random_state=seed, max_iter=1000)
    
    return {"dummy": dummy, "LogisticRegression": logReg}

if __name__ == "__main__":
    print(get_models())