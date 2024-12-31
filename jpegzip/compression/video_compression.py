import os

import cv2 as cv
import numpy as np
from skimage.metrics import mean_squared_error
from utils.file_system import BASE_OUTPUT_DIR, load_video, logger

from compression.image_compression import ImageCompression


class VideoCompression:
    def __init__(self, name: str):
        video, fps = load_video(name)

        self.video: np.ndarray = video
        self.fps: float = fps

        name_compressed = f"{name.split('.')[0]}_compressed.{name.split('.')[1]}"
        self.output_path: str = os.path.join(BASE_OUTPUT_DIR, name_compressed)

    def compress(self) -> float:
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
