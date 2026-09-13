from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def plot_clusters(
    data: pd.DataFrame,
    cluster_column: str = "cluster",
    output_path: Path | None = None,
) -> None:
    """Plot geographic cluster assignments and optionally save the figure."""
    fig, axis = plt.subplots(figsize=(10, 7))
    for cluster, group in data.groupby(cluster_column):
        axis.scatter(
            group["Longitude"],
            group["Latitude"],
            label=f"Cluster {cluster}",
            s=8,
            alpha=0.55,
        )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("California Housing: clusters por ubicación")
    axis.legend(title="Cluster")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_classifier(model, features: pd.DataFrame, labels: pd.Series) -> dict:
    """Return standard classification metrics and the confusion matrix."""
    predictions = model.predict(features)
    return {
        "report": classification_report(labels, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
