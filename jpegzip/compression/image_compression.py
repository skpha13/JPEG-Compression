import numpy as np
from jpegzip.compression.jpeg_compression import JPEGCompression
from jpegzip.utils.image import rgb_to_ycbcr, ycbcr_to_rgb


class ImageCompression:
    @staticmethod
    def compress_rgb(image: np.ndarray) -> np.ndarray:
        """Compress an image using JPEG compression on the YCbCr channels.

        The function compresses the image by converting it to the YCbCr color space
        and applying JPEG compression on the luminance (Y) and chrominance (Cb, Cr)
        channels separately. For grayscale images, only the luminance channel is compressed.

        Parameters
        ----------
        image : np.ndarray
            Input image to be compressed. It can be either a 2D grayscale image or
            a 3D RGB image.

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
        y_channel_compressed = JPEGCompression.decode(JPEGCompression.encode(y_channel, q_method="luminance"))
        cb_channel_compressed = JPEGCompression.decode(JPEGCompression.encode(cb_channel, q_method="chroma")) if cb_channel is not None else None
        cr_channel_compressed = JPEGCompression.decode(JPEGCompression.encode(cr_channel, q_method="chroma")) if cr_channel is not None else None
        # fmt: on

        if image.ndim == 3:
            # fmt: off
            image_ycbcr_compressed = np.stack([y_channel_compressed, cb_channel_compressed, cr_channel_compressed], axis=-1)
            # fmt: on
            rgb_image_compressed = ycbcr_to_rgb(image_ycbcr_compressed)

            return rgb_image_compressed

        return y_channel_compressed
