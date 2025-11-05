import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
import skimage


def generate_horizontal_blend_alpha_mask(shape, window_size=0.5, left=0, right=1):
    # Generates an alpha mask which blends an image into left and right
    height, width, channels = shape
    # Get window coordinates
    window_left = int((0.5 - window_size / 2) * width)
    window_right = int((0.5 + window_size / 2) * width)
    # Generate mask
    return np.repeat(
        np.repeat(
            np.atleast_3d(
                np.concatenate(
                    (
                        np.repeat(left, window_left),
                        np.linspace(left, right, window_right - window_left),
                        np.repeat(right, width - window_right),
                    )
                )
            ),
            height,
            axis=0,
        ),
        channels,
        axis=2,
    )


def alpha_blend(im1, im2, alpha_mask):
    assert im1.shape == im2.shape and im2.shape == alpha_mask.shape
    return (1 - alpha_mask) * im1 + alpha_mask * im2


def save_im(results_dir, components, im):
    io.imsave(
        path.join(results_dir, "_".join(components) + ".jpg"), skimage.img_as_ubyte(im)
    )


def remove_file_extension(file_name):
    segments = file_name.split(".")
    if len(segments) == 1:
        return segments[0]
    return ".".join(segments[:-1])
