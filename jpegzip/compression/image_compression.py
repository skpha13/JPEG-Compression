import logging

import numpy as np
from jpegzip.compression.jpeg_compression import JPEGCompression
from jpegzip.utils.image import rgb_to_ycbcr, ycbcr_to_rgb
from skimage.metrics import mean_squared_error

logger = logging.getLogger(__name__)


class ImageCompression:
    MSE_TOLERANCE: float = 10.0
    MAX_ITERATIONS: int = 15

    @staticmethod
    def compress_rgb(image: np.ndarray, q_factor: float = 1.0) -> np.ndarray:
        """Compress an image using JPEG compression on the YCbCr channels.

        The function compresses the image by converting it to the YCbCr color space
        and applying JPEG compression on the luminance (Y) and chrominance (Cb, Cr)
        channels separately. For grayscale images, only the luminance channel is compressed.

        Parameters
        ----------
        image : np.ndarray
            Input image to be compressed. It can be either a 2D grayscale image or
            a 3D RGB image.

        q_factor : float, optional
            A scaling factor for the quantization matrix used in JPEG compression.

            Higher values of `q_factor` will result in greater compression by increasing
            the quantization step sizes, which may lead to more loss in image quality.
            Conversely, lower values reduce the compression and preserve more image
            details. The default value is 1.

        Returns
        -------
        np.ndarray
            The compressed image. If the input image was a 2D grayscale image,
            only the luminance channel is returned. If the input image was an RGB image,
            the compressed RGB image is returned.

        Raises
        ------
        RuntimeError
            If the image does not have 2 or 3 dimensions, an error is raised.
        """

        if image.ndim != 3 and image.ndim != 2:
            raise RuntimeError(
                f"Invalid image dimensions: {image.ndim}. Expected a 2D grayscale image or a 3D RGB image."
            )

        image_ycbcr = rgb_to_ycbcr(image) if image.ndim == 3 else image

        y_channel = image_ycbcr[:, :, 0] if image.ndim == 3 else image_ycbcr
        cb_channel = image_ycbcr[:, :, 1] if image.ndim == 3 else None
        cr_channel = image_ycbcr[:, :, 2] if image.ndim == 3 else None

        # fmt: off
        y_channel_compressed = JPEGCompression.decode(JPEGCompression.encode(y_channel, q_method="luminance", q_factor=q_factor))
        cb_channel_compressed = JPEGCompression.decode(JPEGCompression.encode(cb_channel, q_method="chroma", q_factor=q_factor)) if cb_channel is not None else None
        cr_channel_compressed = JPEGCompression.decode(JPEGCompression.encode(cr_channel, q_method="chroma", q_factor=q_factor)) if cr_channel is not None else None
        # fmt: on

        if image.ndim == 3:
            # fmt: off
            image_ycbcr_compressed = np.stack([y_channel_compressed, cb_channel_compressed, cr_channel_compressed], axis=-1)
            # fmt: on
            rgb_image_compressed = ycbcr_to_rgb(image_ycbcr_compressed)

            return rgb_image_compressed

        return y_channel_compressed

    @staticmethod
    def compress_to_mse(image: np.ndarray, target_mse: float | None = None, q_factor: float = 1.0) -> np.ndarray:
        """Compresses an image to achieve a specified Mean Squared Error (MSE) using iterative adjustment of the quality factor.

        Parameters
        ----------
        image : np.ndarray
            The input image as a NumPy array.
        target_mse : float, optional
            The target Mean Squared Error (MSE) to achieve after compression. This parameter must be provided and cannot be None.
        q_factor : float, default=1.0
            The initial quality factor for image compression. This value will be iteratively adjusted to meet the target MSE.

        Returns
        -------
        np.ndarray
            The compressed image as a NumPy array that achieves (or is close to) the specified target MSE.

        Raises
        ------
        ValueError
            If `target_mse` is None.
        RuntimeError
            If the target MSE cannot be achieved within the maximum allowed iterations.
        """

        if target_mse is None:
            raise ValueError("Target MSE must be specified and cannot be None. Please provide a valid value.")

        mse: float = 0
        iteration: int = 1
        compressed_image: np.ndarray | None = None

        while np.abs(target_mse - mse) > ImageCompression.MSE_TOLERANCE:
            compressed_image = ImageCompression.compress_rgb(image, q_factor=q_factor)
            mse = mean_squared_error(image, compressed_image)

            logger.info(f" q_factor: {q_factor:10.4f}, mse: {mse:10.4f}")
            q_factor: float = (q_factor * target_mse) / mse

            if iteration > ImageCompression.MAX_ITERATIONS:
                raise RuntimeError(
                    f"Image conversion to the target MSE failed. The best achieved values are: q_factor = {q_factor}. "
                    f"To resolve this, you can try one or more of the following: increase MAX_ITERATIONS, use the provided best values, "
                    f"or increase MSE_TOLERANCE to allow a larger error margin."
                )

            iteration += 1

        return compressed_image
