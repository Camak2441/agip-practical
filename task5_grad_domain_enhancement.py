# Task 5: Gradient domain image enhancement
import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy import signal
import skimage
from skimage import color
import time

from task4_grad_domain import img2grad_field, reconstruct_grad_field

import utils


def piece_wise_linear(points):
    def remap(x):
        return np.interp(x, points[:, 0], points[:, 1])

    return remap


def vec_field_to_mag_field(G):
    return np.sqrt(np.sum(G * G, axis=2))


def remap(G, f):
    mag_field = vec_field_to_mag_field(G)
    new_mag_field = np.vectorize(f)(mag_field)
    new_G = G * np.repeat(
        np.nan_to_num(np.atleast_3d(new_mag_field / mag_field), 0), 2, axis=2
    )
    return new_G


def to_gray_colour_ratio(im):
    im_gray = color.rgb2gray(im)
    return im_gray, im


def to_gray_hsv(im):
    h, s, v = np.unstack(color.rgb2hsv(im), axis=2)
    return v, (h, s)


def to_colour_colour_ratio(im_gray, im):
    im_ratio = im_gray / color.rgb2gray(im)
    imr_color = im * np.repeat(np.atleast_3d(im_ratio), 3, axis=2)
    imr_color = imr_color.clip(0, 1)
    return imr_color


def to_colour_hsv(im_v, hs):
    h, s = hs
    imr_color = color.hsv2rgb(np.stack((h, s, im_v), axis=2))
    imr_color = imr_color.clip(0, 1)
    return imr_color


to_gray = {
    "ratio": to_gray_colour_ratio,
    "hsv": to_gray_hsv,
}

to_colour = {
    "ratio": to_colour_colour_ratio,
    "hsv": to_colour_hsv,
}

enhancement = {
    "lin1": piece_wise_linear(np.array([[0, 0], [0.1, 0.3], [1, 1]])),
    "denoise": piece_wise_linear(
        np.array([[0, 0], [0.04, 0], [0.0401, 0.0401], [1, 1]])
    ),
    "denoise2": piece_wise_linear(
        np.array([[0, 0], [0.02, 0], [0.0201, 0.0201], [1, 1]])
    ),
    "denoise3": piece_wise_linear(
        np.array([[0, 0], [0.004, 0], [0.00401, 0.00401], [1, 1]])
    ),
}


eps = 1e-3


def enhance_image(im, colour_method, enhancement_method, cholesky_decomp=True):
    # Convert to grayscale
    im_gray, recovery_info = to_gray[colour_method](im)

    # Enhance gradient field
    G = img2grad_field(im_gray)
    new_G = remap(G, enhancement[enhancement_method])

    # Reconstruct new grayscale image
    new_im_gray = reconstruct_grad_field(
        new_G,
        1 / (vec_field_to_mag_field(G) + eps),
        im_gray[0, 0],
        im_gray,
        cholesky_decomp=cholesky_decomp,
    )
    # Normalise new gray image to [0, 1]
    new_im_gray = (new_im_gray - np.min(new_im_gray)) / (
        np.max(new_im_gray) - np.min(new_im_gray)
    )

    # Restore colour
    return to_colour[colour_method](new_im_gray, recovery_info)


colour_method = "hsv"
enhancement_method = "lin1"
show_plots = True
results_dir = "results5"

time_enhancement = True


if __name__ == "__main__":
    import sys
    import json

    with open(path.join(results_dir, "timings.json"), "r") as file:
        timings_source = json.load(file)

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
            case "--colour" | "-c":
                index += 1
                if index < len(args):
                    colour_method = args[index]
                    index += 1
                continue
            case "--enhancement" | "-e":
                index += 1
                if index < len(args):
                    enhancement_method = args[index]
                    index += 1
                continue

        im_name = args[index]
        index += 1
        try:
            im_no_extension = utils.remove_file_extension(im_name)

            im = io.imread(path.join("images", im_name))
            im = skimage.img_as_float(im)

            start_time = time.time()
            imr_color = enhance_image(im, colour_method, enhancement_method)
            end_time = time.time()
            cholesky_time = end_time - start_time

            if time_enhancement:
                start_time = time.time()
                enhance_image(
                    im, colour_method, enhancement_method, cholesky_decomp=False
                )
                end_time = time.time()
                sparse_solver_time = end_time - start_time
                timings_source[
                    "_".join([im_no_extension, colour_method, enhancement_method])
                ] = {
                    "width": im.shape[1],
                    "height": im.shape[0],
                    "cholesky_time": cholesky_time,
                    "sparse_solver_time": sparse_solver_time,
                }

            utils.save_im(
                results_dir,
                [im_no_extension, colour_method, "gd", enhancement_method],
                imr_color,
            )

            if show_plots:
                plt.figure(figsize=(9, 3))

                plt.subplot(121)
                plt.title("Original")
                plt.axis("off")
                plt.imshow(im)

                plt.subplot(122)
                plt.title("Enhanced")
                plt.axis("off")
                plt.imshow(imr_color)

                plt.show()
        except Exception as e:
            print(f"Error processing image {im_name}: {e}")
            continue

    with open(path.join(results_dir, "timings.json"), "w") as file:
        json.dump(timings_source, file)
