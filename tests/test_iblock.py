import numpy as np
import pytest
from jpegzip.utils.image import ImageBlockProcessor


class TestIBlock:
    def test_iblocks_reconstruction_simple(self):
        """Test reconstruction of a simple 16x16 image"""

        image = np.arange(256).reshape(16, 16)
        blocks = ImageBlockProcessor.blocks(image)
        reconstructed_image = ImageBlockProcessor.iblocks(blocks)

        np.testing.assert_array_equal(reconstructed_image, image)

    def test_iblocks_reconstruction_padded(self):
        """Test reconstruction with a padded image"""

        image = np.arange(15).reshape(3, 5)
        padded_image = ImageBlockProcessor.pad(image)

        blocks = ImageBlockProcessor.blocks(padded_image)
        reconstructed_image = ImageBlockProcessor.iblocks(blocks)

        np.testing.assert_array_equal(reconstructed_image, padded_image)

    def test_iblocks_reconstruction_empty_blocks(self):
        """Test that iblocks raises an error when given empty blocks"""

        with pytest.raises(RuntimeError):
            ImageBlockProcessor.iblocks(np.array([]))

    def test_iblocks_reconstruction_non_square_blocks(self):
        """Test that iblocks raises an error when the blocks are not square"""

        non_square_blocks = np.ones((2, 2, 8, 7))
        with pytest.raises(RuntimeError):
            ImageBlockProcessor.iblocks(non_square_blocks)

    def test_iblocks_reconstruction_single_block(self):
        """Test reconstruction from a single block"""
        block = np.ones((8, 8))
        blocks = np.array([[block]])

        reconstructed_image = ImageBlockProcessor.iblocks(blocks)

        np.testing.assert_array_equal(reconstructed_image, block)
