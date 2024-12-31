import argparse
import logging
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import scipy
from skimage.metrics import mean_squared_error

from compression.video_compression import VideoCompression
from jpegzip.compression.image_compression import ImageCompression
from jpegzip.utils.plots import plot_compression
from utils.file_system import load_image, save_image

logger = logging.getLogger(__name__)


def compress(image: Optional[np.ndarray] = None) -> np.ndarray:
    if image is None:
        image = scipy.datasets.face()
    compressed_image = ImageCompression.compress_rgb(image)

    plot_compression("RGB Image Compression", image, compressed_image)

    mse = mean_squared_error(image, compressed_image)
    logger.info(f"Mean Squared Error: {mse:.4f}")

    return compressed_image


def compress_to_target_mse(target_mse: float | None = None, image: Optional[np.ndarray] = None) -> np.ndarray:
    if target_mse is None:
        raise ValueError("Target MSE must be specified and cannot be None. Please provide a valid value.")

    if image is None:
        image = scipy.datasets.face()
    compressed_image = ImageCompression.compress_to_mse(image, target_mse=target_mse)

    plot_compression("Target MSE Compression", image, compressed_image)

    mse = mean_squared_error(image, compressed_image)
    logger.info(f" Target MSE: {target_mse:10.4f}, Obtained MSE: {mse:10.4f}")

    return compressed_image


def compress_video() -> None:
    compressor = VideoCompression("sample_video.mp4")
    average_mse = compressor.compress()

    logger.info(f" Average MSE: {average_mse:3.4f}")


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--load", type=str, help="Name of the image to load for compression from folder `input`.")

    subparsers = parser.add_subparsers(dest="operation", help="Choose the compression operation.")

    subparsers.add_parser("compress", help="Compress using the basic RGB compression.")
    compress_to_target_mse_parser = subparsers.add_parser(
        "compress-to-target-mse", help="Compress to a specified target MSE."
    )
    compress_to_target_mse_parser.add_argument(
        "--target-mse", type=float, required=True, help="Target MSE for the compression."
    )
    subparsers.add_parser(
        "compress-video", help="Compress the `sample_video.mp4` located inside the `input` directory."
    )

    return parser


def main() -> None:
    parser = argparse.ArgumentParser(prog="JpegZIP", description="Image Compression CLI")
    parser = add_arguments(parser)

    args = parser.parse_args()

    image = None
    if args.load:
        image = load_image(args.load)

    compressed_image = None
    if args.operation == "compress":
        compressed_image = compress(image)
    elif args.operation == "compress-to-target-mse":
        compressed_image = compress_to_target_mse(args.target_mse, image)
    elif args.operation == "compress-video":
        compress_video()
        return

    image_name = None
    if args.load:
        image_name = f"{args.load.split('.')[0]}_compressed.{args.load.split('.')[1]}"
    else:
        image_name = "raccoon_compressed.png"

    save_image(compressed_image, image_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
