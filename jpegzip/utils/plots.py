import os

import matplotlib.pyplot as plt
import numpy as np

os.makedirs(f"{os.getcwd()}/plots", exist_ok=True)


def save_fig(fig: plt.Figure, plot_name: str) -> None:
    """Saves a matplotlib figure to a specified file location in PNG format.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to save.
    plot_name : str
        The name of the file (without extension) where the figure will be saved.
    """

    fig.savefig(fname=f"plots/{plot_name}.png", format="png")


def plot_compression(plot_name: str, image: np.ndarray, compressed_image: np.ndarray) -> None:
    """Creates and displays a plot comparing the original and compressed images side by side.

    Parameters
    ----------
    plot_name : str
        The title for the plot and the name of the file where the figure will be saved.
    image : np.ndarray
        The original image as a NumPy array.
    compressed_image : np.ndarray
        The compressed image as a NumPy array.
    """

    fig, axs = plt.subplots(2, figsize=(6, 8))

    fig.suptitle(plot_name)

    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(compressed_image)
    axs[1].set_title("Compressed Image")
    axs[1].axis("off")

    save_fig(fig, plot_name)
    plt.show()
