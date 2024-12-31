import logging
import os

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)
BASE_INPUT_DIR = os.path.join(os.getcwd(), "..", "input")
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "..", "output")


def load_image(name: str) -> np.ndarray:
    """Load an image from `input` directory with the specified name.

    Parameters
    ----------
    name : str
        The name of the image file to load. The image file should be located
        in the 'input' directory, relative to the current working directory.

    Returns
    -------
    np.ndarray
        The loaded image as a NumPy array in color format (RGB).

    Raises
    ------
    Exception
        If the image cannot be loaded, an error is logged, and the exception is raised.
    """

    path = os.path.join(BASE_INPUT_DIR, name)

    try:
        image = cv.imread(path, cv.IMREAD_COLOR)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return np.array(image_rgb)
    except Exception as e:
        logger.error(f"Error loading image with path {path}: {e}")
        raise


def save_image(image: np.ndarray, name: str) -> None:
    """Save an image to the `input` directory with the specified name.

    Parameters
    ----------
    image : np.ndarray
        The image to save, as a NumPy array in color format (BGR).

    name : str
        The name of the file to save the image as. The image will be saved in the 'output' directory,
        relative to the current working directory.

    Raises
    ------
    Exception
        If the image cannot be saved, an error is logged, and the exception is raised.
    """

    path = os.path.join(BASE_OUTPUT_DIR, name)

    try:
        image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(path, image_bgr)
    except Exception as e:
        logger.error(f"Error writing image with path {path}: {e}")
        raise
