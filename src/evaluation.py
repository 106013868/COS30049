"""
evaluation.py
splits data, compares models and finds misclassified urls
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.base import clone
import config as cfg
import prepare_data as pdp
from inference import predict

def grouped_split(df, seed=42):
    """
    splits by domain so no domain appears in both train and test
    """
    targets = df[cfg.LABEL_COLUMN]
    groups = df[cfg.DOMAIN_COLUMN]
    # 5 folds, only the first is used so about 20% of domains go to test
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, targets, groups))
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    return train, test

def random_split(df, seed=42):
    """
    stratified 80/20 random split
    """
    train, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[cfg.LABEL_COLUMN])
    
    return train, test

def evaluate_model(model, train, test):
    """
    trains a copy of the model on train, returns scores on test
    """
    model_copy = clone(model)
    train_features = train[cfg.FEATURE_COLUMNS]
    train_labels = train[cfg.LABEL_COLUMN]
    test_features = test[cfg.FEATURE_COLUMNS]
    test_labels = test[cfg.LABEL_COLUMN]

    model_copy.fit(train_features, train_labels)
    pred = model_copy.predict(test_features)

    return score_predictions(test_labels, pred)

def score_predictions(y_true, y_pred):
    """
    returns accuracy, precision, recall, f1 and confusion matrix counts
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # true negative, false positive, false negative, true positive
    t_neg, f_pos, f_neg, t_pos = confusion_matrix(y_true, y_pred).ravel()

    return {
        "acc": accuracy,
        "prec": precision,
        "recall": recall,
        "f1": f1,
        "t_neg": t_neg,
        "f_pos": f_pos,
        "f_neg": f_neg,
        "t_pos": t_pos
    }

def compare_models(models, splits):
    """
    evaluates every model on every split, returns results table
    """
    results = []
    for split_name, (train, test) in splits.items():
        for model_name, m in models.items():
            result = evaluate_model(m, train, test)
            result["split"] = split_name
            result["model"] = model_name
            results.append(result)
    return pd.DataFrame(results)

def get_test_predictions(model, train, test):
    """
    trains a copy of the model, returns test rows with predictions
    """
    fitted = clone(model)
    fitted.fit(train[cfg.FEATURE_COLUMNS], train[cfg.LABEL_COLUMN])

    return predict(fitted, test)

def find_errors(results, n=10, seed=42):
    """
    returns n random false positives and n random false negatives
    """
    # false positive is a safe url predicted malicious
    false_pos = results[(results[cfg.LABEL_COLUMN] == 0) & (results[cfg.PREDICTION_COLUMN] == 1)]
    # false negative is a malicious url predicted safe
    false_neg = results[(results[cfg.LABEL_COLUMN] == 1) & (results[cfg.PREDICTION_COLUMN] == 0)]
    pos_samples = false_pos.sample(n=n, random_state=seed)
    neg_samples = false_neg.sample(n=n, random_state=seed)

    return pos_samples, neg_samples

def run_error_analysis(model, train, test):
    """
    Finds the model's errors on the grouped split and saves the error tables
    """
    results = get_test_predictions(model, train, test)
    print(f"Total false positives: {len(results[(results[cfg.LABEL_COLUMN] == 0) & (results[cfg.PREDICTION_COLUMN] == 1)])}")
    print(f"Total false negatives: {len(results[(results[cfg.LABEL_COLUMN] == 1) & (results[cfg.PREDICTION_COLUMN] == 0)])}")

    fps, fns = find_errors(results, n=10)
    pd.set_option("display.max_colwidth", None)
    cols = [cfg.URL_COLUMN, cfg.LABEL_COLUMN, cfg.PREDICTION_COLUMN, cfg.PROBABILITY_COLUMN, "path_length", "url_length", "digit_count"]
    print("False Positives:")
    print(fps[cols].to_string())
    print("False Negatives:")
    print(fns[cols].to_string())

    all_fps = results[(results[cfg.LABEL_COLUMN] == 0) & (results[cfg.PREDICTION_COLUMN] == 1)]
    top_fp_domains = all_fps[cfg.DOMAIN_COLUMN].value_counts().head(10)
    print("Top 10 domains with false positives:")
    print(top_fp_domains)
    top_domain = top_fp_domains.index[0]
    top_domain_share = top_fp_domains.iloc[0] / len(all_fps) * 100
    print(f"Domain with most false positives: {top_domain}, Share of total false positives: {top_domain_share:.1f}%")
    top10_share = top_fp_domains.sum() / len(all_fps) * 100
    print(f"Share of total false positives for top 10 domains: {top10_share:.1f}%")
    top_domain_in_train = (train[cfg.DOMAIN_COLUMN] == top_domain).sum()

    fps[cols].to_csv(cfg.TABLES_DIR / "errors_false_positives_final.csv", index=False)
    fns[cols].to_csv(cfg.TABLES_DIR / "errors_false_negatives_final.csv", index=False)
    top_fp_domains.to_csv(cfg.TABLES_DIR / "errors_top_fp_domains_final.csv")
    all_fns = results[(results[cfg.LABEL_COLUMN] == 1) & (results[cfg.PREDICTION_COLUMN] == 0)]
    summary = {
        "total_false_positives": len(all_fps),
        "total_false_negatives": len(all_fns),
        "top_domain": top_domain,
        "top_domain_false_positives": top_fp_domains.iloc[0],
        "top_domain_share_percent": top_domain_share,
        "top10_share_percent": top10_share,
        "top_domain_in_training": top_domain_in_train
    }
    pd.DataFrame([summary]).to_csv(cfg.TABLES_DIR / "errors_summary_final.csv", index=False)
    
if __name__ == "__main__":
    print("Running evaluation tests...")
    from models import get_models
    data = pdp.load_data()

    # grouped keeps each domain on one side only, random does not
    splits = {
        "grouped": grouped_split(data),
        "random": random_split(data)
    }
    table = compare_models(get_models(), splits)
    print(table.round(3).to_string())

    # mkdir for evaluation results
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # saved for plots.py
    table.to_csv(cfg.TABLES_DIR / "comparison_final.csv", index=False)

    model = get_models()[cfg.FINAL_MODEL]
    train, test = splits["grouped"]
    run_error_analysis(model, train, test)