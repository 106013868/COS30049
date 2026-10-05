import config as cfg
import pandas as pd
from tldextract import TLDExtract

tld_extract = TLDExtract(suffix_list_urls=())

def load_data():
    if cfg.DATASET_PATH.exists() and cfg.FEATURES_PATH.exists():
        data = load_processed()
    else:
        raise FileNotFoundError("Processed dataset or features not found. Please run the data processing script first.")

    data = add_domain_column(data)

    return data[[cfg.URL_COLUMN, cfg.LABEL_COLUMN, cfg.DOMAIN_COLUMN] + cfg.FEATURE_COLUMNS]

def load_processed(dataset_path=cfg.DATASET_PATH, features_path=cfg.FEATURES_PATH):
    dataset = pd.read_csv(dataset_path)
    features = pd.read_csv(features_path)

    # check row counts match
    if len(dataset) != len(features):
        raise ValueError(f"Row counts do not match: {len(dataset)} != {len(features)}")

    if not dataset[cfg.LABEL_COLUMN].equals(features[cfg.LABEL_COLUMN]):
        raise ValueError("Labels do not match between dataset and features")

    # drop label column from features and combine with dataset
    features = features.drop(columns=[cfg.LABEL_COLUMN])
    combined_df = pd.concat([dataset, features], axis=1)
    
    return combined_df

def extract_domain(url):
    extracted = tld_extract(url)

    if extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    
    return extracted.domain

def add_domain_column(df):
    df[cfg.DOMAIN_COLUMN] = df[cfg.URL_COLUMN].apply(extract_domain)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df[cfg.URL_COLUMN].duplicated().sum())
    print(df[cfg.LABEL_COLUMN].value_counts(normalize=True))
    