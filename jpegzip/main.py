import logging

import scipy
from skimage.metrics import mean_squared_error

from jpegzip.compression.image_compression import ImageCompression
from jpegzip.utils.plots import plot_compression

logger = logging.getLogger(__name__)


def compress_raccoon() -> None:
    image = scipy.datasets.face()
    compressed_image = ImageCompression.compress_rgb(image)

    plot_compression("RGB Image Compression", image, compressed_image)

    mse = mean_squared_error(image, compressed_image)
    logger.info(f"Mean Squared Error: {mse:.4f}")


def compress_raccoon_to_target_mse(target_mse: float | None = None) -> None:
    if target_mse is None:
        raise ValueError("Target MSE must be specified and cannot be None. Please provide a valid value.")

    image = scipy.datasets.face()
    compressed_image = ImageCompression.compress_to_mse(image, target_mse=target_mse)

    plot_compression("Target MSE Compression", image, compressed_image)

    mse = mean_squared_error(image, compressed_image)
    logger.info(f" Target MSE: {target_mse:10.4f}, Obtained MSE: {mse:10.4f}")


def main() -> None:
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compress_raccoon_to_target_mse(300)
