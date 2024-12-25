import scipy
from jpegzip.algorithms.jpeg_compression import JPEGCompression
from matplotlib import pyplot as plt

image = scipy.datasets.ascent()
image_encoded = JPEGCompression.encode(image)
image_decoded = JPEGCompression.decode(image_encoded)

plt.imshow(image, cmap="grey")
plt.show()

plt.imshow(image_decoded, cmap="grey")
plt.show()
