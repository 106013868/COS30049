import numpy as np
import pandas as pd
import config

PROPORTIONS = {0: 0.66, 1: 0.34}

def make_dataset(n_rows=10000, seed=42):
    rng = np.random.default_rng(seed)
    n_domains = n_rows // 4
    domains = make_domains(n_domains)
    df = make_rows(n_rows, domains, rng)
    add_features(df, rng)
    
    return df

# returns a list of n made up domains
def make_domains(n_domains):
    domains = []

    for i in range(n_domains):
        domains.append(f"site{i}.com")

    return domains

def make_rows(n_rows, domains, rng):
    domain = rng.choice(domains, size=n_rows)
    labels = rng.choice(list(PROPORTIONS.keys()), p=list(PROPORTIONS.values()), size=n_rows)
    
    df = pd.DataFrame({config.DOMAIN_COLUMN: domain, config.LABEL_COLUMN: labels})
    
    df[config.URL_COLUMN] = df[config.DOMAIN_COLUMN] + "/page" + df.index.astype(str)

    return df

def add_features(df, rng):
    for f in config.FEATURE_COLUMNS:
        df[f] = rng.random(len(df))
    
    return df

if __name__ == "__main__":
    data = make_dataset()
    