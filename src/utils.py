from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image


def plot_image_grid(
    image_paths: list[Path],
    title: str,
    output_path: Path | None = None,
) -> None:
    """Show up to nine images in a 3x3 grid and optionally save the figure."""
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for axis, image_path in zip(axes.flat, image_paths):
        with Image.open(image_path) as image:
            axis.imshow(image)
            axis.set_title(f"{image_path.name}\n{image.width}x{image.height}", fontsize=9)
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)
    plt.show()
    plt.close(fig)


def plot_predictions(
    image_paths: list[str],
    labels: list[int],
    predictions: list[int],
    confidences: list[float],
    class_names: list[str],
    title: str,
) -> None:
    """Show up to nine images with their real and predicted class (green = hit, red = miss)."""
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.5))
    for axis, path, label, prediction, confidence in zip(
        axes.flat, image_paths, labels, predictions, confidences
    ):
        with Image.open(path) as image:
            axis.imshow(image)
        color = "tab:green" if label == prediction else "tab:red"
        axis.set_title(
            f"Real: {class_names[label]} | Pred: {class_names[prediction]}\n"
            f"confianza {confidence:.1%}",
            fontsize=9,
            color=color,
        )
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_training_history(
    history: dict,
    best_epoch: int | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot loss and accuracy curves for the training and validation sets."""
    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["loss"]) + 1)
    for metric, axis in (("loss", loss_axis), ("accuracy", accuracy_axis)):
        axis.plot(epochs, history[metric], marker="o", markersize=3, label="Train")
        axis.plot(epochs, history[f"val_{metric}"], marker="o", markersize=3, label="Validación")
        if best_epoch is not None:
            axis.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Mejor época ({best_epoch})")
        axis.set_xlabel("Época")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(metric.capitalize())
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)
    plt.show()
    plt.close(fig)
