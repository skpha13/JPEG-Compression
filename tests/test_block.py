import re

import numpy as np
import pytest
from jpegzip.utils.image import ImageBlockProcessor


class TestBlock:
    BLOCK_SIZE: int = 8
    ERROR_MESSAGE: str = f"Image shape is not divisible by {BLOCK_SIZE} on both axis!"

    @staticmethod
    def get_n_blocks(n: int):
        return n // TestBlock.BLOCK_SIZE

    def test_block_8x8(self):
        """Test a small image, exactly 8x8."""
        n = 8
        n_blocks = TestBlock.get_n_blocks(n)

        image_8x8 = np.arange(n**2).reshape(n, n)
        blocks = ImageBlockProcessor.blocks(image_8x8)

        assert blocks.shape == (n_blocks, n_blocks, TestBlock.BLOCK_SIZE, TestBlock.BLOCK_SIZE)

    def test_block_512x512(self):
        """Test larger image, 512x512"""
        n = 512
        n_blocks = TestBlock.get_n_blocks(n)

        image_24x24 = np.arange(n**2).reshape(n, n)
        blocks = ImageBlockProcessor.blocks(image_24x24)

        assert blocks.shape == (n_blocks, n_blocks, TestBlock.BLOCK_SIZE, TestBlock.BLOCK_SIZE)

    def test_non_square_8x16(self):
        """Test image shape 8x16"""

        image_non_square = np.ones((8, 16))
        blocks = ImageBlockProcessor.blocks(image_non_square)

        assert blocks.shape == (1, 2, TestBlock.BLOCK_SIZE, TestBlock.BLOCK_SIZE)

    def test_block_contents_16x16(self):
        """Test content and order of 16x16 image"""
        n = 16

        image_16x16 = np.arange(n**2).reshape(n, n)
        blocks_generated = ImageBlockProcessor.blocks(image_16x16)

        block_ground_truth = np.array(
            [
                [128, 129, 130, 131, 132, 133, 134, 135],
                [144, 145, 146, 147, 148, 149, 150, 151],
                [160, 161, 162, 163, 164, 165, 166, 167],
                [176, 177, 178, 179, 180, 181, 182, 183],
                [192, 193, 194, 195, 196, 197, 198, 199],
                [208, 209, 210, 211, 212, 213, 214, 215],
                [224, 225, 226, 227, 228, 229, 230, 231],
                [240, 241, 242, 243, 244, 245, 246, 247],
            ]
        )

        assert np.array_equal(block_ground_truth, blocks_generated[1][0])

    def test_invalid_image_shape(self):
        # Test case where image shape is not divisible by 8

        image = np.ones((7, 8))
        with pytest.raises(RuntimeError, match=TestBlock.ERROR_MESSAGE):
            ImageBlockProcessor.blocks(image)

        image = np.ones((8, 7))
        with pytest.raises(RuntimeError, match=TestBlock.ERROR_MESSAGE):
            ImageBlockProcessor.blocks(image)

        image = np.ones((7, 7))
        with pytest.raises(RuntimeError, match=TestBlock.ERROR_MESSAGE):
            ImageBlockProcessor.blocks(image)

    def test_empty_image(self):
        # Test case where the image is empty (0x0 shape)
        image = np.zeros((0, 0))

        with pytest.raises(RuntimeError, match=re.escape(f"Image is empty! Image shape: {image.shape}.")):
            ImageBlockProcessor.blocks(image)
