"""
clustering.py
clusters malicious urls with kmeans and summarises each cluster
"""
import pandas as pd
import config as cfg
from prepare_data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def prepare_cluster_data(df):
    """
    filters to malicious rows and scales features, returns filtered df and scaled features
    """
    # filter to malicious rows
    df = df[df[cfg.LABEL_COLUMN] == 1].copy()
    features = df[cfg.FEATURE_COLUMNS]
    scaled_features = StandardScaler().fit_transform(features)

    return df, scaled_features

def choose_k(scaled_features, k_values, seed=42):
    """
    fits kmeans for each k, returns inertia and silhouette score per k
    """
    results = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        kmeans.fit(scaled_features)
        # silhouette is scored on a 10000 row sample to keep it fast
        score = silhouette_score(scaled_features, kmeans.labels_, sample_size=10000, random_state=seed)

        results.append({"k": k, "inertia": kmeans.inertia_, "silhouette_score": score})

    return pd.DataFrame(results)

def fit_clusters(scaled_features, k, seed=42):
    """
    fits kmeans with chosen k and returns cluster labels
    """
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(scaled_features)

    return kmeans.labels_

def describe_clusters(df, scaled_features):
    """
    returns raw and standardised feature means per cluster
    """
    # raw means
    raw_table = df.groupby(df[cfg.CLUSTER_COLUMN])[cfg.FEATURE_COLUMNS].mean()
    # overall mean added as a comparison row
    overall = df[cfg.FEATURE_COLUMNS].mean().to_frame("overall").T
    raw_table = pd.concat([raw_table, overall], axis=0)

    # standardised means
    scaled = pd.DataFrame(scaled_features, columns=cfg.FEATURE_COLUMNS, index=df.index)
    standardised_table = scaled.groupby(df[cfg.CLUSTER_COLUMN]).mean()

    return raw_table, standardised_table

def cluster_examples(df, n=5, seed=42):
    """
    returns n random urls per cluster and prints non-ip examples from cluster 1
    """
    examples = df.groupby(cfg.CLUSTER_COLUMN).sample(n=n, random_state=seed)
    cols = [cfg.URL_COLUMN, cfg.CLUSTER_COLUMN, "url_length", "subdomain_count", "has_ip"]

    # cluster 1 is mostly ip urls, so check what the non-ip ones look like
    non_ip = df[(df[cfg.CLUSTER_COLUMN] == 1) & (df["has_ip"] == 0)]
    print("Cluster 1, non-IP examples:")
    print(non_ip.sample(5, random_state=seed)[cols].to_string())

    return examples[cols]

if __name__ == "__main__":
    df = load_data()
    df, scaled_features = prepare_cluster_data(df)
    # fit k = 2 to 8 to compare elbow and silhouette
    k_scores = choose_k(scaled_features, range(2, 9))
    print(k_scores.round(3).to_string())

    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # saved for plots.py
    k_scores.to_csv(cfg.TABLES_DIR / "clustering_k_scores_final.csv", index=False)

    # k = 3 chosen, k = 2 only separates ip urls
    df[cfg.CLUSTER_COLUMN] = fit_clusters(scaled_features, 3)
    # cluster sizes and label composition
    print("Cluster sizes:")
    print(df[cfg.CLUSTER_COLUMN].value_counts(normalize=True))
    print("Label composition:")
    print(pd.crosstab(df[cfg.CLUSTER_COLUMN], df[cfg.LABEL_COLUMN]))
    # feature means per cluster
    raw_table, standardised_table = describe_clusters(df, scaled_features)
    print("Raw means:")
    print(raw_table.T.round(2).to_string())
    print("Standardised differences:")
    print(standardised_table.T.round(2).to_string())
    raw_table.to_csv(cfg.TABLES_DIR / "clustering_raw_means_final.csv")
    standardised_table.to_csv(cfg.TABLES_DIR / "clustering_standardised_final.csv")

    # show full urls in the examples
    pd.set_option("display.max_colwidth", None)
    examples = cluster_examples(df, n=8)
    examples = examples.sort_values(cfg.CLUSTER_COLUMN)
    print(examples.to_string())
    examples.to_csv(cfg.TABLES_DIR / "clustering_examples_final.csv")