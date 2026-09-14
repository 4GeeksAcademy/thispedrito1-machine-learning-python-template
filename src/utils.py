from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_training_history(history: dict, output_path: Path | None = None) -> None:
    """Plot loss and accuracy curves for the training and validation sets."""
    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))
    for metric, axis in (("loss", loss_axis), ("accuracy", accuracy_axis)):
        axis.plot(history[metric], label="Train")
        axis.plot(history[f"val_{metric}"], label="Validación")
        axis.set_xlabel("Época")
        axis.set_title(metric.capitalize())
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)
    plt.show()
    plt.close(fig)
