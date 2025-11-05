# Task 4: Gradient domain reconstruction

import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import skimage.metrics
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sksparse.cholmod import cholesky
from scipy import signal
import skimage
import time

import utils


def img2grad_field(img):
    """
    Return a gradient field for a greyscale image
    The function returns image [height,width,2], where the last dimension selects partial derivates along x or y
    """
    # img must be a greyscale image
    sz = img.shape
    G = np.zeros([sz[0], sz[1], 2])
    # Gradients along x-axis
    G[:, :, 0] = signal.convolve2d(
        img, np.array([1, -1, 0]).reshape(1, 3), "same", boundary="symm"
    )
    # Gradients along y-axis
    G[:, :, 1] = signal.convolve2d(
        img, np.array([1, -1, 0]).reshape(3, 1), "same", boundary="symm"
    )
    return G


def reconstruct_grad_field(G, w, v_00, img, cholesky_decomp=True):
    """Reconstruct a (greyscale) image from a gradient field
    G - gradient field, for example created with img2grad_field
    w - weight assigned to each gradient
    v_00 - the value of the first pixel
    """
    sz = G.shape[:2]
    N = sz[0] * sz[1]

    # Gradient operators as sparse matrices
    o1 = np.ones((N, 1))
    B = np.concatenate(
        (-o1, np.concatenate((np.zeros((sz[0], 1)), o1[: N - sz[0]]), 0)), 1
    )
    B[N - sz[0] : N, 0] = 0
    Ogx = sparse.spdiags(
        B.transpose(), [0, sz[0]], N, N
    )  # Forward difference operator along x

    B = np.concatenate((-o1, np.concatenate((np.array([[0]]), o1[0 : N - 1]), 0)), 1)
    B[sz[0] - 1 :: sz[0], 0] = 0
    B[sz[0] :: sz[0], 1] = 0
    Ogy = sparse.spdiags(
        B.transpose(), [0, 1], N, N
    )  # Forward difference operator along y

    # Separating the x and y gradients
    Gx, Gy = np.unstack(G, axis=2)
    # Converting to column vectors in column-major order
    Gx = Gx.T.reshape(N, 1)
    Gy = Gy.T.reshape(N, 1)

    # Generating the sparse vectors
    sGx = sparse.coo_array(Gx)
    sGy = sparse.coo_array(Gy)

    # Producing the weighting matrix
    w = w.T.reshape(N)
    W = sparse.diags(w)

    # Producing the zeroth pixel matching constraint
    C = np.concatenate(([1], np.zeros(N - 1)))
    sC = sparse.coo_array(C.reshape(N, 1))
    CC = sparse.diags(C)

    # Calculating A and b
    A = Ogx.T @ W @ Ogx + Ogy.T @ W @ Ogy + CC
    b = Ogx.T @ W @ sGx + Ogy.T @ W @ sGy + sC * v_00

    # Solve
    if cholesky_decomp:
        factor = cholesky(A)
        im = factor(b).toarray()
    else:
        im = sparse.linalg.spsolve(A, b)

    # Return the image in the correct shape
    return np.reshape(im, (sz[1], sz[0])).T


results_dir = "results4"
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
            im_no_extension = utils.remove_file_extension(im_name)

            im = io.imread(path.join("images", im_name), as_gray=True)
            im = skimage.img_as_float(im)

            # Calculate gradient field
            G = img2grad_field(im)
            Gm = np.sqrt(np.sum(G * G, axis=2))

            w = 1 / (Gm + 0.0001)  # To avoid pinching artefacts

            imr = reconstruct_grad_field(G, w, im[0, 0], im).clip(0, 1)

            # Save results
            utils.save_im(results_dir, [im_no_extension, "original"], im)
            utils.save_im(results_dir, [im_no_extension, "reconstructed"], imr)
            utils.save_im(
                results_dir, [im_no_extension, "difference"], (imr - im + 1) / 2
            )

            if show_plots:
                plt.figure(figsize=(9, 3))

                plt.subplot(131)
                plt.title("Original")
                plt.axis("off")
                plt.imshow(im, cmap="gray")

                plt.subplot(132)
                plt.title("Reconstructed")
                plt.axis("off")
                plt.imshow(imr, cmap="gray")

                PSNR_recon = skimage.metrics.peak_signal_noise_ratio(im, imr)
                plt.subplot(133)
                plt.title(f"Difference: PSNR={PSNR_recon:.2f} dB")
                plt.axis("off")
                plt.imshow(imr - im, vmin=0, vmax=0.5)

                plt.show()

        except Exception as e:
            print(f"Error processing image {im_name}: {e}")
            continue
