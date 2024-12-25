import scipy
from jpegzip.utils.image import rgb_to_ycbcr, ycbcr_to_rgb
from skimage.metrics import mean_squared_error


class TestConversion:
    """A test class to validate the accuracy and integrity of RGB to YCbCr color conversion
    and its inverse (YCbCr to RGB) using Mean Squared Error (MSE) as a metric.

    Attributes:
    -----------
    MSE_THRESHOLD : float
       A constant value representing the acceptable threshold for the Mean Squared Error (MSE)
       between the original and reconstructed images. Default value is 10.0.
    """

    MSE_THRESHOLD: float = 10.0

    def test_rgb_ycbcr_mse(self):
        """Validates the RGB to YCbCr and YCbCr to RGB conversion processes.

        Notes
        -----
        - Checks that all reconstructed RGB values fall within the valid range [0, 255].
        - Computes the Mean Squared Error (MSE) between the original and reconstructed images.
        - Asserts that the MSE is less than the defined threshold (`MSE_THRESHOLD`).
        """

        image = scipy.datasets.face()

        ycbcr = rgb_to_ycbcr(image)
        rgb = ycbcr_to_rgb(ycbcr)

        assert (rgb >= 0).all() and (rgb <= 255).all()

        mse = mean_squared_error(image, rgb)

        assert mse < TestConversion.MSE_THRESHOLD
