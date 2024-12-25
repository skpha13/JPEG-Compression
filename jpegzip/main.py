import matplotlib.pyplot as plt
import scipy
from skimage.metrics import mean_squared_error

from jpegzip.compression.image_compression import ImageCompression
from jpegzip.utils.plots import save_fig


def compress_rgb_image() -> None:
    image = scipy.datasets.face()
    compressed_image = ImageCompression.compress_rgb(image)

    fig, axs = plt.subplots(2, figsize=(6, 8))
    plot_name = "RGB Image Compression"

    fig.suptitle(plot_name)

    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(compressed_image)
    axs[1].set_title("Compressed Image")
    axs[1].axis("off")

    mse = mean_squared_error(image, compressed_image)
    print(f"Mean Squared Error: {mse}")

    save_fig(fig, plot_name)
    plt.show()


def main() -> None:
    pass


if __name__ == "__main__":
    compress_rgb_image()
