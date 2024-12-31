# JpegZIP

**JpegZIP** is a Python-based package designed to have image and video
compression using the **JPEG algorithm**.

## [Documentation](./docs/documentation.md)

## [Developer's Guide](./docs/developer_guide.md)

## How It Works

The core approach follows the methodology outlined in the [Wikipedia article](https://en.wikipedia.org/wiki/JPEG#JPEG_codec_example)
on JPEG, which describes the JPEG codec in detail.

In addition to the standard JPEG pipeline, **JpegZIP** introduces advanced customizations like
targeted MSE compression and video frame-by-frame processing.

## Compression Based on Target MSE

### Description

Achieve precise control over the compression process by setting a target Mean Squared Error (MSE).
This feature compresses an image iteratively until the specified MSE difference is reached,
balancing file size and quality according to your requirements.

### Implementation

The algorithm begins with an initial compression factor, `Q_FACTOR`, set to `1`. This value is used to scale the coefficients in the quantizable matrix. The goal is to iteratively adjust the `Q_FACTOR` to achieve a target Mean Squared Error (MSE) for a compressed image compared to the original image.

### Key Steps

1. **Calculate the Adjusted `Q-Factor`**

Using proportional reasoning, compute a new candidate compression factor, `Q_FACTOR_NEW`, that would bring the current MSE closer to the target MSE:

```math
\text{Q-FACTOR-NEW} = (\text{Q-FACTOR} * \text{TARGET-MSE}) / \text{CURRENT-MSE}
```

2. **Determine the Adjustment `Step`**

Calculate the difference between the new and current `Q-Factor` values, referred to as `STEP`:

```math
\text{STEP} = \text{Q-FACTOR-NEW} - \text{Q-FACTOR}
```

3. **Adjust the `Q-Factor` Gradually**

Move the current `Q-Factor` toward `Q_FACTOR_NEW` by a fraction of `STEP`, controlled by a predefined constant `FACTOR_RATE` (e.g., `0.5`):

```math
\text{Q-FACTOR} = \text{Q-FACTOR} + \text{STEP} * \text{FACTOR-RATE}
```

4. **Iterate Until Convergence or Stop**

Repeat the above steps until the difference between the current MSE and the target MSE is within an acceptable tolerance, `MAX_TOLERANCE`, or after the `MAX_ITERATIONS` has been reached.

## Video Compression

In the context of video compression, **intra-frame compression** refers to the process of compressing each individual frame independently, without considering dependencies or similarities with other frames in the video. Each frame is treated as a standalone image, and compression algorithms are applied directly to the pixel data within that frame.

After compression, the video is reconstructed by assembling all the individually compressed frames in sequence.

## Observations

The docstring documentation follows the **NumPy style guide** and was generated with the assistance of **ChatGPT**. Additionally, certain sections of the README were enhanced and refined using this AI tool.

## References

[Cristian Rusu's Signal Processing Lectures](https://cs.unibuc.ro/~crusu/ps/lectures.html)

[Wikipeadia JPEG codec example](https://en.wikipedia.org/wiki/JPEG#JPEG_codec_example)

[Wikipeadia YcbCr Conversion](https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion)

[Wikipedia Video Compression](https://en.wikipedia.org/wiki/Video_coding_format)

[OpenCV Getting Started with Video](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)

[Understanding Color Spaces and Color Space Conversion](https://www.mathworks.com/help/images/understanding-color-spaces-and-color-space-conversion.html)

[Matrix Block Division](https://stackoverflow.com/questions/55175042/how-to-divide-images-into-8x8-blocks-and-merge-them-back-using-opencv)

[Sample Video](https://www.sample-videos.com/)

[Sample Image](https://sketchfab.com/3d-models/firewatch-fan-art-8609caf1cd8c452eb7b6d4ca4228fcd0)
