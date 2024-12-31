import logging
import os

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)
BASE_INPUT_DIR = os.path.join(os.getcwd(), "input")
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "output")


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


def load_video(name: str) -> tuple[np.ndarray, float]:
    """Load a video file, extract all frames in RGB format, and return them along with the video FPS.

    Parameters
    ----------
    name : str
        The name of the video file to load. The path will be constructed using the `BASE_INPUT_DIR` directory.

    Returns
    -------
    tuple
        A tuple containing:
        - `frames_array` : numpy.ndarray
            A 4D NumPy array of shape `(num_frames, height, width, 3)` where each frame is an RGB image.
        - `fps` : float
            The frames per second (FPS) of the video.

    Raises
    ------
    SystemExit
        If the video cannot be opened, the function will log an error and exit the program.
    """

    path = os.path.join(BASE_INPUT_DIR, name)
    video = cv.VideoCapture(path)
    fps = video.get(cv.CAP_PROP_FPS)

    if not video.isOpened():
        logger.error(f"Error loading video with path: {path}")
        exit()

    frames = []
    while True:
        ret, frame = video.read()

        if not ret:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(rgb_frame)

    video.release()

    frames_array = np.array(frames)
    return frames_array, fps
