import numpy as np


class ImageBlockProcessor:
    BLOCK_SIZE: int = 8

    @staticmethod
    def pad(image: np.ndarray) -> np.ndarray:
        """Pad the input image to the nearest multiple of BLOCK_SIZE in both dimensions (height and width).

        This function adds padding to the input image such that the resulting image
        has dimensions that are multiples of BLOCK_SIZE. The padding is applied symmetrically
        to the top, bottom, left, and right sides of the image.

        Parameters
        ----------
        image : np.ndarray
            A 2D NumPy array representing the input image to be padded.

        Returns
        -------
        np.ndarray
            A 2D NumPy array representing the padded image with dimensions
            that are multiples of BLOCK_SIZE in both height and width.

        Notes
        -----
        - If the input image's dimensions are already multiples of BLOCK_SIZE, no padding will be applied.
        - The padding is applied using a constant value of 0 (black padding) around the edges.
        """

        rows, cols = image.shape

        pad_rows = (
            ImageBlockProcessor.BLOCK_SIZE - (rows % ImageBlockProcessor.BLOCK_SIZE)
        ) % ImageBlockProcessor.BLOCK_SIZE
        pad_cols = (
            ImageBlockProcessor.BLOCK_SIZE - (cols % ImageBlockProcessor.BLOCK_SIZE)
        ) % ImageBlockProcessor.BLOCK_SIZE

        top = pad_rows // 2
        bottom = pad_rows - top
        left = pad_cols // 2
        right = pad_cols - left

        padded_image = np.pad(image, ((top, bottom), (left, right)), mode="constant")

        return padded_image

    @staticmethod
    def is_image_shape_divisible_block_size(image: np.ndarray):
        """Checks if the shape of the input image is divisible by BLOCK_SIZE on both axes.

        Parameters
        ----------
        image : np.ndarray
            The input image as a 2D NumPy array.

        Raises
        ------
        RuntimeError
            If the image is empty (either rows or columns are 0), raises an error indicating the image is empty.
        RuntimeError
            If the shape of the image is not divisible by `BLOCK_SIZE` on both axes, raises an error.
        """

        rows, cols = image.shape

        if rows == 0 or cols == 0:
            raise RuntimeError(f"Image is empty! Image shape: {image.shape}.")

        if rows % ImageBlockProcessor.BLOCK_SIZE != 0 or cols % ImageBlockProcessor.BLOCK_SIZE != 0:
            raise RuntimeError(f"Image shape is not divisible by {ImageBlockProcessor.BLOCK_SIZE} on both axis!")

    @staticmethod
    def blocks(image: np.ndarray) -> np.ndarray:
        """Split an image into non-overlapping BLOCK_SIZE x BLOCK_SIZE blocks.

        Parameters
        ----------
        image : np.ndarray
           Input image to be split. Should be a 2D array where each element represents a pixel in the image.
           The shape of the image must be divisible by BLOCK_SIZE in both dimensions.

        Returns
        -------
        np.ndarray
           A 4D array of shape (n, m, BLOCK_SIZE, BLOCK_SIZE),
           where each element in the matrix is a 2D block of size BLOCK_SIZE x BLOCK_SIZE extracted from the input image.

        Notes
        -----
        The function assumes the image shape is divisible by BLOCK_SIZE (both rows and columns). If the shape is not divisible
        by BLOCK_SIZE, an error will be raised.
        """

        ImageBlockProcessor.is_image_shape_divisible_block_size(image)

        rows, cols = image.shape

        # how many BLOCK_SIZE x BLOCK_SIZE non-intersecting blocks can fit in the rows and columns provided
        n_row_split = rows // ImageBlockProcessor.BLOCK_SIZE
        n_col_split = cols // ImageBlockProcessor.BLOCK_SIZE

        # np.split returns a view into the array
        image_copy = image.copy()
        # split rows in segments of size BLOCK_SIZE
        image_sliced = np.split(image_copy, n_row_split, axis=0)
        # split the resulting rows in columns of size BLOCK_SIZE, thus resulting in an array of shape: (n_row_split, n_col_split, BLOCK_SIZE, BLOCK_SIZE)
        image_blocks = np.array([np.split(img_slice, n_col_split, axis=1) for img_slice in image_sliced])

        return image_blocks

    @staticmethod
    def iblocks(blocks: np.ndarray) -> np.ndarray:
        """Reconstruct an image from its block decomposition.

        This function takes a 4D array of image blocks and reconstructs the original image by
        concatenating the blocks. The blocks must be of size (BLOCK_SIZE, BLOCK_SIZE).

        Parameters
        ----------
        blocks : numpy.ndarray
            A 4D array of shape (n, m, BLOCK_SIZE, BLOCK_SIZE) where each sub-array represents
            a block of the image.

        Returns
        -------
        numpy.ndarray
            The reconstructed image as a 2D numpy array.

        Raises
        ------
        RuntimeError
            If the input array does not have four dimensions, or if the blocks are not square.
        """

        if blocks.ndim != 4 or blocks.shape[2] != blocks.shape[3]:
            raise RuntimeError(
                f"Input blocks are not of shape (n, m, BLOCK_SIZE, BLOCK_SIZE). As a result the image cannot be reconstructed."
            )

        # reconstruct rows by concatenating blocks along the columns
        reconstructed_rows = [np.concatenate(row_blocks, axis=1) for row_blocks in blocks]
        # reconstruct the entire image by concatenating the rows
        reconstructed_image = np.concatenate(reconstructed_rows, axis=0)

        return reconstructed_image


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to YCbCr color space.

    Parameters
    ----------
    image : np.ndarray
        A numpy array of shape (H, W, 3) representing an RGB image.
        Each pixel is a 3-element array with red, green, and blue components.

    Returns
    -------
    np.ndarray
        A numpy array of shape (H, W, 3) representing the corresponding YCbCr image.
        Each pixel is a 3-element array with Y, Cb, and Cr components.
    """

    transform_matrix = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ]
    )
    bias = np.array([0, 128, 128])

    ycbcr_image = image @ transform_matrix.T + bias

    return np.clip(ycbcr_image, 0, 255).astype(np.uint8)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a YCbCr image to RGB color space.

    Parameters
    ----------
    image : np.ndarray
        A numpy array of shape (H, W, 3) representing a YCbCr image.
        Each pixel is a 3-element array with Y, Cb, and Cr components.

    Returns
    -------
    np.ndarray
        A numpy array of shape (H, W, 3) representing the corresponding RGB image.
        Each pixel is a 3-element array with red, green, and blue components.
    """

    itransform_matrix = np.array(
        [
            [1, 0, 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0],
        ]
    )
    bias = np.array([0, 128, 128])

    ycbcr_image = image.astype(np.float32) - bias
    rgb_image = ycbcr_image @ itransform_matrix.T

    return np.clip(rgb_image, 0, 255).astype(np.uint8)
