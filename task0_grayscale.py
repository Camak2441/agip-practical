# This is an example on how to read and display an image, convert between colour spaces.
# This file does not need to be included in the submission.

import os.path as path
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import skimage.util as util
import skimage.color as color

# change the img_name to the image you want to use
img_name = "naples1.jpg"

# read the image
im = io.imread(path.join("images", img_name))

plt.axis("off")
io.imshow(im)
plt.show()

# The image is represented as a numpy object with shape (height x width x channels)
# It usually contains RGB channels stored as 8bit unsigned integer values.
print("The dimension of the image is {}".format(im.shape))
print("The type of the image representation is {}".format(im.dtype))
print("The max/min pixel values are ({}, {})".format(im.max(), im.min()))

# Here rgb2grey calculate the luma map of the original image
grey_im = color.rgb2gray(im)
print("The dimension of the grey image is {}".format(grey_im.shape))
print("The type of the image representation is {}".format(grey_im.dtype))
print("The max/min pixel values are ({}, {})".format(grey_im.max(), grey_im.min()))
plt.axis("off")
io.imshow(grey_im)
plt.show()


# Here we show the gray-scale channel in other colour spaces
hsv_im = color.rgb2hsv(im)
luv_im = color.rgb2luv(im)
yuv_im = color.rgb2yuv(im)

plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.title("grey")
plt.axis("off")
plt.imshow(grey_im, cmap="gray")

plt.subplot(222)
plt.title("hsv")
plt.axis("off")
plt.imshow(hsv_im[:, :, 2], cmap="gray")  # v channel

plt.subplot(223)
plt.title("luv")
plt.axis("off")
plt.imshow(luv_im[:, :, 0] / 100, cmap="gray")  # l channel

plt.subplot(224)
plt.title("yuv")
plt.axis("off")
plt.imshow(yuv_im[:, :, 0], cmap="gray")  # y channel
plt.show()
