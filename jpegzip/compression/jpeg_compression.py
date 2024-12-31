from typing import Literal

import numpy as np
import scipy
from jpegzip.utils.image import ImageBlockProcessor
from scipy.fft import idctn


class JPEGCompression:
    """Implements JPEG-like image compression and decompression using the Discrete Cosine Transform (DCT)
    and quantization.

    Attributes
    ----------
    Q_LUMINANCE : list[list[int]]
        Quantization matrix used for compression. It is a 8x8 matrix containing
        standard JPEG luminance quantization values.

    Q_CHROMA: list[list[int]]
        Quantization matrix used for compression. It is a 8x8 matrix containing
        standard JPEG chroma quantization values.

    PIXEL_MEAN : int
        Value subtracted from image pixels to center them around zero.

    Q_DOWNSAMPLING : int
        Downsampling factor applied during preprocessing to reduce high-frequency noise.
    """

    Q_LUMINANCE: list[list[int]] = [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]

    Q_CHROMA: list[list[int]] = [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]

    PIXEL_MEAN: int = 128
    Q_DOWNSAMPLING: int = 10

    @staticmethod
    def encode(
        image: np.ndarray, q_method: Literal["luminance", "chroma"] = "luminance", q_factor: float = 1.0
    ) -> np.ndarray:
        """Compresses an input image using JPEG-like encoding.

        The process includes downsampling, centering pixel values to zero, block-wise DCT,
        and quantization using the specified quantization matrix.

        Parameters
        ----------
        image : np.ndarray
            Input image represented as a 2D numpy array.

        q_method : Literal["luminance", "chroma"], optional
            The quantization method to use during compression.

            If "luminance", the luminance quantization matrix will be used,
            which is typically used for the brightness component of the image.

            If "chroma", the chrominance quantization matrix will be used,
            which is typically used for the color components (U, V) of the image.

            The default is "luminance".

        q_factor : float, optional
            A scaling factor for the quantization matrix used in JPEG compression.

            Higher values of `q_factor` will result in greater compression by increasing
            the quantization step sizes, which may lead to more loss in image quality.
            Conversely, lower values reduce the compression and preserve more image
            details. The default value is 1.

        Returns
        -------
        np.ndarray
            Encoded image represented as a 2D numpy array.
            The output includes quantized DCT coefficients for each 8x8 block.
        """
        Q = []
        if q_method == "luminance":
            Q = JPEGCompression.Q_LUMINANCE
        elif q_method == "chroma":
            Q = JPEGCompression.Q_CHROMA
        Q = np.astype(np.array(Q), np.float32)
        Q = q_factor * Q

        x = image.copy()

        # preprocess image by downsampling and centering pixels to 0
        x = JPEGCompression.Q_DOWNSAMPLING * np.round(x / JPEGCompression.Q_DOWNSAMPLING)
        x = np.subtract(x, JPEGCompression.PIXEL_MEAN)

        x = ImageBlockProcessor.pad(x)
        x_blocks = ImageBlockProcessor.blocks(x)

        # apply dctn on the last 2 axes (8x8 blocks)
        y_dctn_blocks = scipy.fft.dctn(x_blocks, axes=(-2, -1))
        y_dctn_quantized = Q * np.round(y_dctn_blocks / Q)

        y = ImageBlockProcessor.iblocks(y_dctn_quantized)

        return y

    @staticmethod
    def decode(image: np.ndarray) -> np.ndarray:
        """Decompresses an encoded image using JPEG-like decoding.

        The process includes block-wise IDCT, dequantization, and re-centering pixel values
        back to their original range.

        Parameters
        ----------
        image : np.ndarray
            Encoded image represented as a 2D numpy array,
            containing quantized DCT coefficients.

        Returns
        -------
        np.ndarray
            Decoded image represented as a 2D numpy array.
            The output is an approximation of the original image before encoding.
        """

        y_blocks = ImageBlockProcessor.blocks(image)

        y_idctn_blocks = idctn(y_blocks, axes=(-2, -1))
        y_idctn_blocks = np.round(np.add(y_idctn_blocks, JPEGCompression.PIXEL_MEAN))

        y = ImageBlockProcessor.iblocks(y_idctn_blocks)

        return y
