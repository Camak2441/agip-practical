# Task 3: Pyramid blending

import os.path as path
import skimage.io as io

import math
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt

from task2_alpha_blending import alpha_blending

import utils


def gausspyr_reduce(x, kernel_a=0.4):
    """
    Filter and subsample the image x. Used to create consecutive levels of the Gaussian pyramid [1]. Can process both grayscale and colour images.
    x - image to subsample
    kernel_a - the coefficient of the kernel

    returns an image that is half the size of the input x.

    [1] Burt, P., & Adelson, E. (1983). The Laplacian Pyramid as a Compact Image Code. IEEE Transactions on Communications, 31(4), 532-540. https://doi.org/10.1109/TCOM.1983.1095851
    """
    # Kernel
    K = np.array([0.25 - kernel_a / 2, 0.25, kernel_a, 0.25, 0.25 - kernel_a / 2])
    hK = K.reshape(1, -1)
    vK = K.reshape(-1, 1)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # Add an extra dimension if grayscale
    y = np.zeros(
        [math.ceil(x.shape[0] / 2), math.ceil(x.shape[1] / 2), x.shape[2]]
    )  # Store the result in this array
    for cc in range(x.shape[2]):  # for each colour channel
        # Step 1: filter rows
        # Step 2: subsample rows (skip every second column)
        temp = sp.signal.convolve2d(x[:, :, cc], hK, mode="same", boundary="symm")[
            :, ::2
        ]
        # Step 3: filter columns
        # Step 4: subsample columns (skip every second row)
        y[:, :, cc] = sp.signal.convolve2d(temp, vK, mode="same", boundary="symm")[
            ::2, :
        ]

    return np.squeeze(y)  # remove an extra dimension for grayscale images


def gausspyr_expand(x, sz=None, kernel_a=0.4):
    """
    Double the size and interpolate using Gaussian pyramid kernel [1]. Can process both grayscale and colour images.
    x - image to upsample
    sz - [height, width] of the generated image. Necessary if one of the dimensions of the upsampled image is odd.
    kernel_a - the coefficient of the kernel

    returns an image that is  double the size or the size of sz of the input x.

    [1] Burt, P., & Adelson, E. (1983). The Laplacian Pyramid as a Compact Image Code. IEEE Transactions on Communications, 31(4), 532–540. https://doi.org/10.1109/TCOM.1983.1095851
    """

    # Kernel is multipled by 2 to preserve energy when increasing the resolution
    K = 2 * np.array([0.25 - kernel_a / 2, 0.25, kernel_a, 0.25, 0.25 - kernel_a / 2])

    if sz is None:
        sz = (x.shape[0] * 2, x.shape[1] * 2)

    x = x.reshape(x.shape[0], x.shape[1], -1)  # Add an extra dimension if grayscale
    y = np.zeros([sz[0], sz[1], x.shape[2]])
    for cc in range(x.shape[2]):  # for each colour channel
        y_a = np.zeros((x.shape[0], sz[1]))
        y_a[:, ::2] = x[:, :, cc]
        y_a = sp.signal.convolve2d(
            y_a, K.reshape(1, -1), mode="same", boundary="symm"
        )  # filter rows
        y[::2, :, cc] = y_a
        y[:, :, cc] = sp.signal.convolve2d(
            y[:, :, cc], K.reshape(-1, 1), mode="same", boundary="symm"
        )  # filter columns

    return np.squeeze(y)  # remove an extra dimension for grayscale images


class laplacian_pyramid:
    @staticmethod
    def gaussian_decompose(img, levels=-1):
        """
        Decompose img into a Gaussian pyramid.
        levels: how many levels should be created (including the base band). When the default (-1) value is used, the maximum possible number of levels is created.
        """

        # The maximum number of levels we can have
        max_levels = math.floor(math.log2(min(img.shape[0], img.shape[1])))

        assert levels < max_levels

        if levels == -1:
            levels = max_levels  # Use max_levels by default

        curr_img = img
        gauss_pyramid = [curr_img]

        for _ in range(levels - 1):
            curr_img = gausspyr_reduce(curr_img)
            gauss_pyramid.append(curr_img)

        return gauss_pyramid

    @staticmethod
    def decompose(img, levels=-1):
        """
        Decompose img into a Laplacian pyramid.
        levels: how many levels should be created (including the base band). When the default (-1) value is used, the maximum possible number of levels is created.
        """

        # The maximum number of levels we can have
        max_levels = math.floor(math.log2(min(img.shape[0], img.shape[1])))

        assert levels < max_levels

        if levels == -1:
            levels = max_levels  # Use max_levels by default

        gauss_pyramid = laplacian_pyramid.gaussian_decompose(img, levels)

        laplace_pyramid = gauss_pyramid.copy()

        for level in range(levels - 1):
            larger = gauss_pyramid[level]
            smaller = gauss_pyramid[level + 1]
            expanded = gausspyr_expand(smaller, larger.shape)
            laplace_pyramid[level] = larger - expanded

        return laplace_pyramid

    @staticmethod
    def reconstruct(pyramid):
        """
        Combine the levels of the Laplacian pyramid to reconstruct an image.
        """

        img = None

        img = pyramid[-1]
        for level in range(len(pyramid) - 2, -1, -1):
            img = gausspyr_expand(img, pyramid[level].shape)
            img += pyramid[level]

        return img


def pyramid_blending(im1, im2, levels=4, window_size=0.3):
    assert im1.shape == im2.shape

    pyramid1 = laplacian_pyramid.decompose(im1, levels)
    pyramid2 = laplacian_pyramid.decompose(im2, levels)
    alpha_mask = utils.generate_horizontal_blend_alpha_mask(im1.shape, window_size)
    alpha_pyramid = laplacian_pyramid.gaussian_decompose(alpha_mask, levels)

    assert len(pyramid1) == len(pyramid2) and len(pyramid2) == len(alpha_pyramid)

    blended_pyramid = pyramid1.copy()

    curr_window_size = window_size / 2 ** len(pyramid1)
    for i in range(len(pyramid1)):
        blended_pyramid[i] = utils.alpha_blend(
            pyramid1[i],
            pyramid2[i],
            utils.generate_horizontal_blend_alpha_mask(
                pyramid1[i].shape, window_size=curr_window_size
            ),
        )
        curr_window_size *= 2

    blended_im = laplacian_pyramid.reconstruct(blended_pyramid)
    return blended_im


results_dir = "results3"
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

        im_names = args[index].split(",")
        index += 1
        try:
            im_name_left = im_names[0]
            im_name_right = im_names[1]
            im_no_extension_left = utils.remove_file_extension(im_name_left)
            im_no_extension_right = utils.remove_file_extension(im_name_right)

            # Part 1: Laplacian pyramid decomposition
            im = io.imread(path.join("images", im_name_left))
            im = util.img_as_float(im[:, :, :3])
            im = color.rgb2gray(im)

            pyramid = laplacian_pyramid.decompose(im, levels=4)

            plt.figure(figsize=(3 * len(pyramid), 3))
            grid = len(pyramid) * 10 + 121
            for i, layer in enumerate(pyramid):
                plt.subplot(grid + i)
                plt.title("level {}".format(i))
                plt.axis("off")
                if i == len(pyramid) - 1:
                    plt.imshow(color.gray2rgb(layer))
                else:
                    plt.imshow(layer)

            plt.subplot(grid + len(pyramid))
            plt.title("reconstruction")
            plt.axis("off")
            im_reconstructed = laplacian_pyramid.reconstruct(pyramid)
            plt.imshow(np.clip(im_reconstructed, 0, 1))

            plt.subplot(grid + len(pyramid) + 1)
            plt.title("differences")
            plt.axis("off")
            plt.imshow(np.abs(im - im_reconstructed))

            plt.savefig(
                path.join(results_dir, im_no_extension_left + "_laplacian_pyramid.jpg")
            )
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Part 2: Pyramid blending
            im_left = io.imread(path.join("images", im_name_left))
            im_left = util.img_as_float(im_left[:, :, :3])
            im_right = io.imread(path.join("images", im_name_right))
            im_right = util.img_as_float(im_right[:, :, :3])
            im_alpha_blend = alpha_blending(
                im_left,
                im_right,
                utils.generate_horizontal_blend_alpha_mask(
                    im_left.shape, window_size=0.25
                ),
            ).clip(0, 1)
            im_pyramid_blend = pyramid_blending(
                im_left, im_right, window_size=0.25
            ).clip(0, 1)

            utils.save_im(
                results_dir,
                [im_no_extension_left, im_no_extension_right, "alpha", "blend"],
                im_alpha_blend,
            )
            utils.save_im(
                results_dir,
                [im_no_extension_left, im_no_extension_right, "pyramid", "blend"],
                im_pyramid_blend,
            )

            if show_plots:
                plt.figure(figsize=(15, 12))
                plt.subplot(221)
                plt.title("left image")
                plt.axis("off")
                plt.imshow(im_left)

                plt.subplot(222)
                plt.title("right image")
                plt.axis("off")
                plt.imshow(im_right)

                plt.subplot(223)
                plt.title("alpha blend")
                plt.axis("off")
                plt.imshow(im_alpha_blend)

                plt.subplot(224)
                plt.title("pyramid blend")
                plt.axis("off")
                plt.imshow(im_pyramid_blend)
                plt.show()

        except Exception as e:
            print(f"Error processing image {im_names}: {e}")
            continue
