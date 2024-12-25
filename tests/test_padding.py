import numpy as np
from jpegzip.utils.image import ImageBlockProcessor


class TestPadding:
    def test_pad_8x8_no_padding_needed(self):
        """Test case where dimensions are already divisible by 8."""

        image = np.zeros((16, 16))
        padded_image = ImageBlockProcessor.pad(image)

        assert padded_image.shape == image.shape

    def test_pad_8x8_padding_needed(self):
        """Test case for a grayscale image requiring padding."""

        image = np.zeros((15, 30))
        padded_image = ImageBlockProcessor.pad(image)

        # should be padded to 16x32
        assert padded_image.shape == (16, 32)
        assert np.all(padded_image[0, :] == 0)

    def test_pad_8x8_odd_dimensions(self):
        """Test case for odd dimensions."""

        image = np.zeros((17, 25))
        padded_image = ImageBlockProcessor.pad(image)

        assert padded_image.shape == (24, 32)
