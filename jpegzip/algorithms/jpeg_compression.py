import numpy as np
import scipy
from jpegzip.utils.image import ImageBlockProcessor


class JPEGCompression:
    Q = [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
    PIXEL_MEAN = 128
    BLOCK_SIZE = 8

    @staticmethod
    def encode(image: np.ndarray) -> np.ndarray:
        x = image.copy()
        # preprocess image by centering pixels to 0
        x = np.subtract(x, JPEGCompression.PIXEL_MEAN)

        # check image is divisible by 8
        rows, cols = x.shape
        if rows % JPEGCompression.BLOCK_SIZE != 0 or cols % JPEGCompression.BLOCK_SIZE != 0:
            x = ImageBlockProcessor.pad(x)

        print(x)

    @staticmethod
    def decode(image: np.ndarray) -> np.ndarray:
        pass
