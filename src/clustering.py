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
    overall = df[cfg.FEATURE_COLUMNS].mean().to_frame("overall").T
    raw_table = pd.concat([raw_table, overall], axis=0)
    
    scaled = pd.DataFrame(scaled_features, columns=cfg.FEATURE_COLUMNS, index=df.index)
    standardised_table = scaled.groupby(df[cfg.CLUSTER_COLUMN]).mean()

    return raw_table, standardised_table

def cluster_examples(df, n=5, seed=42):
    examples = df.groupby(cfg.CLUSTER_COLUMN).sample(n=n, random_state=seed)
    cols = [cfg.URL_COLUMN, cfg.CLUSTER_COLUMN, "url_length", "subdomain_count", "has_ip"]

    non_ip = df[(df[cfg.CLUSTER_COLUMN] == 1) & (df["has_ip"] == 0)]
    print("Cluster 1, non-IP examples:")
    print(non_ip.sample(5, random_state=42)[cols].to_string())

    return examples[cols]

if __name__ == "__main__":
    df = load_data()
    df, scaled_features = prepare_cluster_data(df)
    k_scores = choose_k(scaled_features, range(2, 9))
    print(k_scores.round(3).to_string())

    cfg.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    k_scores.to_csv(cfg.EVAL_DIR / "clustering_k_scores_final.csv", index=False)
    
    df[cfg.CLUSTER_COLUMN] = fit_clusters(scaled_features, 3)
    print("Cluster sizes:")
    print(df[cfg.CLUSTER_COLUMN].value_counts(normalize=True))
    print("Label composition:")
    print(pd.crosstab(df[cfg.CLUSTER_COLUMN], df[cfg.LABEL_COLUMN]))
    raw_table, standardised_table = describe_clusters(df, scaled_features)
    print("Raw means:")
    print(raw_table.T.round(2).to_string())
    print("Standardised differences:")
    print(standardised_table.T.round(2).to_string())
    raw_table.to_csv(cfg.EVAL_DIR / "clustering_raw_means_final.csv")
    standardised_table.to_csv(cfg.EVAL_DIR / "clustering_standardised_final.csv")

    pd.set_option("display.max_colwidth", None)
    examples = cluster_examples(df, n=8)
    examples = examples.sort_values(cfg.CLUSTER_COLUMN)
    print(examples.to_string())
    examples.to_csv(cfg.EVAL_DIR / "clustering_examples_final.csv")

    
    
    """
    cluster = fit_clusters(scaled_features, k=3)
    df[cfg.CLUSTER_COLUMN] = cluster

    raw_table, standardised_table = describe_clusters(df, scaled_features)
    print("Raw Means Table:")
    print(raw_table.round(2).to_string())
    print("\nStandardised Means Table:")
    print(standardised_table.round(2).to_string())

    # check: for each cluster, what percentage of its URLs contain ://.
    for cluster_id in df[cfg.CLUSTER_COLUMN].unique():
        cluster_df = df[df[cfg.CLUSTER_COLUMN] == cluster_id]
        count_with_colon_slash_slash = cluster_df[cfg.URL_COLUMN].str.contains("://").sum()
        percentage_with_colon_slash_slash = (count_with_colon_slash_slash / len(cluster_df)) * 100
        print(f"Cluster {cluster_id}: {percentage_with_colon_slash_slash:.2f}% of URLs contain '://'.")
    """