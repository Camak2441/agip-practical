# Task 2: Alpha blending

import os.path as path
import skimage.io as io

import numpy as np
import scipy as sp
from skimage import color
from skimage import util

import matplotlib.pyplot as plt

import utils


def hard_blending(im_left, im_right, alpha_mask):
    """
    return an image that consist of the left-half of im_left
    and right-half of im_right
    """
    assert im_right.shape == im_left.shape
    bw_mask = (alpha_mask > 0.5).astype(int)
    new_im = utils.alpha_blend(im_left, im_right, bw_mask)
    return new_im


def alpha_blending(im_left, im_right, alpha_mask):
    """
    return a new image that smoothly combines im1 and im2
    im_left: np.array image of the dimensions: height x width x channels; values: 0-1
    im_right: np.array same dim as im_left
    """
    assert im_right.shape == im_left.shape
    return utils.alpha_blend(im_left, im_right, alpha_mask)


results_dir = "results2"
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
            im_no_extensions = list(map(utils.remove_file_extension, im_names))
            
            im_left = io.imread(path.join("images", im_names[0]))
            im_left = util.img_as_float(im_left[:, :, :3])
            im_right = io.imread(path.join("images", im_names[1]))
            im_right = util.img_as_float(im_right[:, :, :3])

            mask = utils.generate_horizontal_blend_alpha_mask(
                im_left.shape, window_size=0.2
            )

            if len(im_names) > 2:
                mask = io.imread(path.join("images", im_names[2]))
                mask = util.img_as_float(mask[:, :, :3])
                mask = color.rgb2gray(mask)

            hard_blend_im = hard_blending(im_left, im_right, mask).clip(0, 1)
            alpha_blend_im = alpha_blending(im_left, im_right, mask).clip(0, 1)

            if len(im_names) > 2:
                utils.save_im(
                    results_dir,
                    [im_no_extensions[0], im_no_extensions[1], im_no_extensions[2], "hard_blend"],
                    hard_blend_im,
                )
                utils.save_im(
                    results_dir,
                    [im_no_extensions[0], im_no_extensions[1], im_no_extensions[2], "alpha_blend"],
                    alpha_blend_im,
                )
                utils.save_im(
                    results_dir,
                    [im_no_extensions[0], im_no_extensions[1], im_no_extensions[2], "mask"],
                    mask,
                )
            else:
                utils.save_im(
                    results_dir, [im_no_extensions[0], im_no_extensions[1], "hard_blend"], hard_blend_im
                )
                utils.save_im(
                    results_dir,
                    [im_no_extensions[0], im_no_extensions[1], "alpha_blend"],
                    alpha_blend_im,
                )
                utils.save_im(
                    results_dir,
                    [im_no_extensions[0], im_no_extensions[1], "mask"],
                    mask,
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
                plt.title("hard blending")
                plt.axis("off")
                plt.imshow(hard_blend_im)

                plt.subplot(224)
                plt.title("alpha blending")
                plt.axis("off")
                plt.imshow(alpha_blend_im)

                plt.show()

        except Exception as e:
            print(f"Error processing image {im_names}: {e}")
            continue
