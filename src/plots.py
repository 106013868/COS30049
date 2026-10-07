import pandas as pd
import matplotlib.pyplot as plt
import config as cfg

def plot_k_choice(chosen_k=3):
    k_scores = pd.read_csv(cfg.TABLES_DIR / "clustering_k_scores_final.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(k_scores["k"], k_scores["inertia"], marker="o")
    ax1.set_title("Elbow method")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.axvline(chosen_k, linestyle="--", color="grey")
    ax1.set_xticks(k_scores["k"])

    ax2.plot(k_scores["k"], k_scores["silhouette_score"], marker="o")
    ax2.set_title("Silhouette score")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette score")
    ax2.axvline(chosen_k, linestyle="--", color="grey")
    ax2.set_xticks(k_scores["k"])

    fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / "clustering_k_choice.png", dpi=200)

    plt.close(fig)

def plot_model_f1(model=cfg.FINAL_MODEL):
    results = pd.read_csv(cfg.TABLES_DIR / "comparison_final.csv")
    results = results[results["split"] == "grouped"]
    results = results.sort_values("f1")

    colours = [ "tab:blue" if name == model else "lightgrey" for name in results["model"] ]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(results["model"], results["f1"], color=colours)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("F1 score (grouped split)")
    ax.set_title("Model comparison")
    ax.set_xlim(0, 0.75)

    fig.tight_layout()
    fig.savefig(cfg.PLOTS_DIR / "model_f1_comparison.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_k_choice()
    plot_model_f1()