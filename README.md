## PhishLens
COS30049 Computing Technology Innovation Project, Assignment 2.

PhishLens is an offline phishing URL detector that scores how trustworthy a link is from its text alone, without ever visiting the site.

- **Repository:** https://github.com/106013868/COS30049

## Team

| Name | Student ID |
|---|---|
| Tim Johnson | 106013868 |
| Raza Syed | 104041595 |
| Anthony Duero | 104561608 |

## Setup

From the project root, create a virtual environment and install the packages:

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Then install the packages:

```bash
pip install -r requirements.txt
```

This installs numpy, pandas, scikit-learn, joblib, tldextract and matplotlib at the versions the project was tested with.

## Data

PhishLens uses two public Kaggle datasets.

| Dataset | Source | Raw file | Processed files |
|---|---|---|---|
| 1. Malicious URLs dataset | [kaggle.com/datasets/sid321axn/malicious-urls-dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) | `malicious_phish.csv` | `processed_dataset.csv`, `processed_features.csv` |
| 2. Phishing Site URLs | [kaggle.com/datasets/taruntiwarihp/phishing-site-urls](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) | `phishing_site_urls.csv` | `processed_test_dataset.csv`, `processed_test_features.csv` |

**Raw files are not in the repo.** Download them from Kaggle and place them in `datasets/` only if you want to rerun `data_processing.py`.

**Processed files are in the repo**, so every other script runs without downloading anything:

| File | Size |
|---|---|
| `datasets/processed_dataset.csv` | 39.1 MB |
| `datasets/processed_features.csv` | 16.0 MB |
| `datasets/processed_test_dataset.csv` | 28.0 MB |
| `datasets/processed_test_features.csv` | 12.6 MB |

## Usage

Run the scripts in this order. Evaluation, training, clustering and plots need the processed CSVs in `datasets/`. `plots.py` also needs the tables from `evaluation.py` and `clustering.py`.

### 1. `data_processing.py`

```bash
cd datasets
python ../src/data_processing.py
```

Cleans `malicious_phish.csv` (drops missing and duplicate URLs, strips `http(s)://`, `www.` and trailing slashes, maps labels to 0/1), extracts the 10 URL features, and writes `processed_dataset.csv` and `processed_features.csv` into the folder it is run from, so run it from `datasets/`.

### 2. `evaluation.py`

```bash
python src/evaluation.py
```

Compares all four models on a grouped split and a random split, prints the scores, and saves them to `evaluation/tables/comparison_final.csv`. Can take a few minutes.

### 3. `train.py`

```bash
python src/train.py
```

Trains the final model (balanced HistGradientBoosting) on the full dataset, saves it to `models/hist_gradient_boosting.joblib`, and prints predictions for 20 sample URLs.

### 4. `clustering.py`

```bash
python src/clustering.py
```

Runs k-means on the malicious URLs only (k = 2 to 8, final k = 3) and saves the k scores, cluster means and example URLs to `evaluation/tables/clustering_*_final.csv`.

### 5. `plots.py`

```bash
python src/plots.py
```

Draws the k-choice chart and the model F1 comparison chart and saves them to `evaluation/plots/`.

## Project structure

```
.
├── README.md
├── requirements.txt
├── datasets/
├── evaluation/
│   ├── tables/
│   ├── plots/
├── models/
└── src/
    ├── config.py                 paths, column names, feature list
    ├── data_processing.py        cleaning and feature extraction
    ├── prepare_data.py           loads processed CSVs, adds domain column
    ├── models.py                 candidate models
    ├── evaluation.py             grouped and random split comparison
    ├── train.py                  train, save, load, predict
    ├── clustering.py             k-means on malicious URLs
    └── plots.py                  charts from the saved tables
```
