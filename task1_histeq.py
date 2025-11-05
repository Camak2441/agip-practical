# Task 1: Image enhancement

import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util

import matplotlib.pyplot as plt

import utils

# Useful functions: np.bincount and np.cumsum


def equalise_hist(image, bin_count=256):
    """
    Perform histogram equalization on an image and return as a new image.

    Arguments:
    image -- a numpy array of shape height x width, dtype float, range between 0 and 1
    bin_count -- how many bins to use
    """

    height, width = image.shape
    bins, _ = np.histogram(image, bins=bin_count, range=(0, 1))

    cumhist = np.cumsum(bins)

    def remap(brightness):
        return cumhist[min(int(brightness * bin_count), bin_count - 1)]

    vremap = np.vectorize(remap)
    new_image = vremap(image) / (height * width)

    return new_image


def he_per_channel(img):
    # Get channels, and normalise each one in term
    channels = np.unstack(img, axis=2)
    new_channels = [equalise_hist(channel) for channel in channels]
    # Restack equalised channels into image
    new_img = np.stack(new_channels, axis=2)
    return new_img


def he_colour_ratio(img):
    # Get the grayscale image
    gray_img = color.rgb2gray(img)
    equalised_gray_img = equalise_hist(gray_img)
    # Get the brightness ratio between the old gray image and the normalised gray image
    ratio_img = np.nan_to_num(equalised_gray_img / gray_img, 0)
    new_img = img * np.repeat(np.atleast_3d(ratio_img), 3, axis=2)
    # Ensure the values in the image are clamped in range
    new_img = new_img.clip(0, 1)
    return new_img


def he_hsv(img):
    # Get the hsv image
    hsv_img = color.rgb2hsv(img)
    h, s, v = np.unstack(hsv_img, axis=2)
    # Equalise the value channel
    equalised_v = equalise_hist(v)
    new_hsv_img = np.stack((h, s, equalised_v), axis=2)
    # Convert the hsv image back into an rgb image
    new_img = color.hsv2rgb(new_hsv_img)
    return new_img


results_dir = "results1"
show_plots = True

if __name__ == "__main__":
    import sys

    args = sys.argv
    index = 1

    while index < len(args):
        match args[index]:
            case "--show-plots" | "-s":
                show_plots = True
                index += 1
                continue
            case "--no-show-plots" | "-n":
                show_plots = False
                index += 1
                continue

        im_name = args[index]
        index += 1
        try:
            img_no_extension = utils.remove_file_extension(im_name)

            test_im = io.imread(path.join("images", im_name))

            test_im_gray = color.rgb2gray(test_im)
            test_im_gray_eq = equalise_hist(test_im_gray)
            utils.save_im(results_dir, [img_no_extension, "gray"], test_im_gray)
            utils.save_im(
                results_dir, [img_no_extension, "gray", "eq"], test_im_gray_eq
            )

            if show_plots:
                plt.figure(figsize=(15, 5))
                plt.subplot(121)
                plt.title("Original image")
                plt.axis("off")
                plt.imshow(test_im_gray, cmap="gray")

                plt.subplot(122)
                plt.title("Histogram equalised image")
                plt.axis("off")
                plt.imshow(test_im_gray_eq, cmap="gray")

                plt.show()

            test_im = io.imread(path.join("images", im_name))
            test_im = util.img_as_float(test_im)
            test_im_eq_per_channel = he_per_channel(test_im)
            utils.save_im(
                results_dir,
                [img_no_extension, "per_channel", "eq"],
                test_im_eq_per_channel,
            )
            test_im_eq_ratio = he_colour_ratio(test_im)
            utils.save_im(
                results_dir, [img_no_extension, "ratio", "eq"], test_im_eq_ratio
            )
            test_im_eq_hsv = he_hsv(test_im)
            utils.save_im(results_dir, [img_no_extension, "hsv", "eq"], test_im_eq_hsv)

            if show_plots:
                plt.figure(figsize=(15, 12))

                plt.subplot(221)
                plt.title("Original image")
                plt.axis("off")
                plt.imshow(test_im)

                plt.subplot(222)
                plt.title("Each channel processed seperately")
                plt.axis("off")
                plt.imshow(test_im_eq_per_channel)

                plt.subplot(223)
                plt.title("Grey-scale + colour ratio")
                plt.axis("off")
                plt.imshow(test_im_eq_ratio)

                plt.subplot(224)
                plt.title("Processed V in HSV")
                plt.axis("off")
                plt.imshow(test_im_eq_hsv)

                plt.show()
        except Exception as e:
            print(f"Error processing image {im_name}: {e}")
            continue
