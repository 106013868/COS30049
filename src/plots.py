import pandas as pd
import matplotlib.pyplot as plt
import config as cfg

def plot_k_choice(chosen_k=3):
    k_scores = pd.read_csv(cfg.EVAL_DIR / "clustering_k_scores_final.csv")
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
    fig.savefig(cfg.EVAL_DIR / "clustering_k_choice.png", dpi=200)

    plt.close(fig)

if __name__ == "__main__":
    plot_k_choice()