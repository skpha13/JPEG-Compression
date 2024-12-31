import os

import cv2 as cv
import numpy as np
from skimage.metrics import mean_squared_error
from utils.file_system import BASE_OUTPUT_DIR, load_video, logger

from compression.image_compression import ImageCompression


class VideoCompression:
    """A class to handle video compression by compressing each frame using the
    ImageCompression utility and saving the compressed video.

    Parameters
    ----------
    name : str
        The name of the video file to be compressed, from the `input` directory.

    Attributes
    ----------
    video : numpy.ndarray
        A 4D numpy array representing the video frames with shape (frames, height, width, channels).
    fps : float
        The frames per second of the input video.
    output_path : str
        The file path for saving the compressed video.
    """

    def __init__(self, name: str):
        """Initializes the VideoCompression class by loading the video, extracting
        its frames per second (fps), and setting up the output path for the compressed video.

        Parameters
        ----------
        name : str
            The name of the video file to be compressed that is located in the `input` directory.
        """

        video, fps = load_video(name)

        self.video: np.ndarray = video
        self.fps: float = fps

        name_compressed = f"{name.split('.')[0]}_compressed.{name.split('.')[1]}"
        self.output_path: str = os.path.join(BASE_OUTPUT_DIR, name_compressed)

    def compress(self) -> float:
        """Compresses the video frame by frame using the ImageCompression utility.
        Saves the compressed video to the output path and calculates the average mean
        squared error (MSE) for the compression.

        Returns
        -------
        float
            The average mean squared error (MSE) between the original and compressed frames.

        Notes
        -----
        Each frame is compressed using the `ImageCompression.compress_rgb` method.
        The video is saved in MP4 format with the codec 'mp4v'.
        """

        frames, height, width, _ = self.video.shape

        # video codec for mp4 file
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out_video = cv.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        mses: list[float] = []

        for index, frame in enumerate(self.video):
            compressed_frame = ImageCompression.compress_rgb(frame)
            compressed_frame_bgr = cv.cvtColor(compressed_frame, cv.COLOR_RGB2BGR)
            out_video.write(compressed_frame_bgr)

            mses.append(mean_squared_error(frame, compressed_frame))

            logger.info(f" Current frame: {index:4}/{frames:4}")

        out_video.release()

        return np.mean(mses)
