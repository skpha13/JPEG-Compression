# Documentation

## Install

To get started, install the package by following these steps:

```bash
# clone repository
git clone https://github.com/skpha13/JPEG-Compression.git

# enter the directory
cd JPEG-Compression

# install all required dependencies
pip install .
```

## Command Line Interface (CLI)

This package provides a simple and intuitive CLI for compressing images.

### Summary of Key Commands

| Command                                       | Description                                                                  |
|-----------------------------------------------|------------------------------------------------------------------------------|
| `-h`                                          | Displays help information.                                                   |
| `--load <image_name>`                         | Loads a custom image for compression from the `input` directory.             |
| `compress`                                    | Compresses the currently loaded image or the default raccoon image.          |
| `compress-to-target-mse --target-mse <value>` | Compresses the image to the specified target MSE.                            |
| `compress-video`                              | Compresses the video named `sample_video.mp4` inside the `input` directory.  |


### Help

To see the available commands and options, use:

```bash
python -m jpegzip.main -h
```

### Loading an Image

To load an image for compression, place the image in the `input` directory, then run:

```bash
python -m jpegzip.main --load sample_image.png
```

> [!WARNING]
> Note: The `--load` command is mandatory and must always come before any other commands when using custom images.

### Example: Compressing a Custom Image

To compress an image immediately after loading:

```bash
python -m jpegzip.main --load sample_image.png compress
```

### Compressing the Default Image

If no image is provided, the tool will automatically select the default `raccoon` image from the SciPy dataset (`scipy.datasets.face`).

To compress the default image:

```bash
python -m jpegzip.main compress
```

### Compress to a Target MSE

To compress an image while targeting a specific Mean Squared Error (MSE), use the `compress-to-target-mse` command.

#### Example: Compressing with Target MSE

To compress the default image with a target MSE of 100:

```bash
python -m jpegzip.main compress-to-target-mse --target-mse 100
```

#### Example: Compressing a Custom Image with Target MSE

To compress a custom image (`sample_image.png`) with a target MSE of 100:

```bash
python -m jpegzip.main --load sample_image.png compress-to-target-mse --target-mse 100
```

### Compressing a Video

To compress the `sample_video.mp4` inside the `input` directory.

```bash
python -m jpegzip.main compress-video
```

> [!WARNING]
> If you want to compress a custom video you will need to place it in the `input` directory
and rename it to `sample_video.mp4`.

## Notes

> [!NOTE]
> - Ensure all images you wish to process are placed in the `input` directory.
> - The compressed images will be outputted to the `output` directory.
> - Always load custom images using `--load` before running any compression commands.
