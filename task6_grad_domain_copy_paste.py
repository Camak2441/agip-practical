# Task 6: Gradient domain copy & paste

import os.path as path
import skimage.io as io
import numpy as np
from skimage.util import img_as_uint
import scipy as sp
from skimage import color
from skimage import util
import skimage.filters as filters
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy import signal
import skimage
from scipy.interpolate import interp1d
from skimage import color
from scipy import ndimage
from sksparse.cholmod import cholesky
import time

from task4_grad_domain import img2grad_field

import utils

show_plots = True
results_dir = "results6"
debug = False

if __name__ == "__main__":
    import sys

    args = sys.argv
    index = 1

    while index < len(args):
        match args[index]:
            case "--debug" | "-d":
                debug = True
                index += 1
                continue
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

            # Read background image
            bg = io.imread(path.join("images", im_names[0]))[:, :, :3]
            bg = skimage.img_as_float(bg)
            # Read foreground image
            fg = io.imread(path.join("images", im_names[1]))
            fg = skimage.img_as_float(fg)
            # Calculate alpha mask
            mask = (fg[:, :, 3] > 0.5).astype(int)
            fg = fg[:, :, :3]  # drop alpha channel

            # Gradient domain copy paste

            # Get shape of the image
            sz = bg.shape

            # Get the number of masked pixels (pixels to be calculated)
            N = np.sum(mask)
            # Produce map from pixel location to pixel number
            im_to_masked_map = np.reshape(np.cumsum(mask) - 1, shape=(sz[0], sz[1]))
            # Produce map from pixel number to pixel location
            masked_to_im_map = np.zeros(shape=(N, 2))

            # Get the edge, using erosion and logical operators
            edge = np.logical_and(np.logical_not(ndimage.binary_erosion(mask)), mask)
            # Get the number of edge pixels
            K = np.sum(edge)
            # Produce map from pixel location to edge number
            edge_map = np.reshape(np.cumsum(edge) - 1, shape=(sz[0], sz[1]))

            # Produce map from pixel number to pixel location
            for row in range(sz[0]):
                for col in range(sz[1]):
                    if mask[row, col]:
                        masked_to_im_map[im_to_masked_map[row, col], :] = [row, col]

            masked_to_im_map = np.array(masked_to_im_map, dtype=int)

            ccs = np.unstack(fg, axis=2)
            Gs = [np.unstack(img2grad_field(channel), axis=2) for channel in ccs]
            Gxs = np.stack([G[0] for G in Gs], axis=2)
            Gys = np.stack([G[1] for G in Gs], axis=2)

            # Prepare arrays for sparse data matrices
            OgxData = []
            OgxRowInds = []
            OgxColInds = []
            OgyData = []
            OgyRowInds = []
            OgyColInds = []
            EData = []
            ERowInds = []
            EColInds = []
            TE = np.zeros(shape=(K, 3))
            masked_Gxs = np.zeros(shape=(N, 3))
            masked_Gys = np.zeros(shape=(N, 3))

            # Get data in sparse data matrices
            # Uses for loops, since the mask shape is arbitrary
            for pixel in range(N):
                row, col = masked_to_im_map[pixel]
                if col + 1 < sz[1] and mask[row, col + 1]:
                    # Forward x difference
                    OgxData.append(-1)
                    OgxRowInds.append(pixel)
                    OgxColInds.append(pixel)
                    OgxData.append(1)
                    OgxRowInds.append(pixel)
                    OgxColInds.append(im_to_masked_map[row, col + 1])
                if row + 1 < sz[0] and mask[row + 1, col]:
                    # Forward y difference
                    OgyData.append(-1)
                    OgyRowInds.append(pixel)
                    OgyColInds.append(pixel)
                    OgyData.append(1)
                    OgyRowInds.append(pixel)
                    OgyColInds.append(im_to_masked_map[row + 1, col])
                if mask[row, col]:
                    # Gradient field
                    masked_Gxs[im_to_masked_map[row, col]] = Gxs[row, col]
                    masked_Gys[im_to_masked_map[row, col]] = Gys[row, col]
                if edge[row, col]:
                    # Edge constraint
                    EData.append(1)
                    ERowInds.append(edge_map[row, col])
                    EColInds.append(pixel)
                    TE[edge_map[row, col]] = bg[row, col]

            # Produce sparse matrices
            Ogx = sparse.csr_array((OgxData, (OgxRowInds, OgxColInds)), shape=(N, N))
            Ogy = sparse.csr_array((OgyData, (OgyRowInds, OgyColInds)), shape=(N, N))
            E = sparse.csr_array((EData, (ERowInds, EColInds)), shape=(K, N))

            # Calculate A and b
            A = Ogx.T @ Ogx + Ogy.T @ Ogy + E.T @ E
            bs = np.unstack(Ogx.T @ masked_Gxs + Ogy.T @ masked_Gys + E.T @ TE, axis=1)

            # Solve for image
            factor = cholesky(A)
            new_masked = np.stack([factor(b) for b in bs], axis=1)

            I_dest = bg.copy()

            # Paste the solved pixels over background image
            for i in range(N):
                row, col = masked_to_im_map[i]
                I_dest[row, col] = new_masked[i]

            # Clip result to [0, 1]
            I_dest = I_dest.clip(0, 1)

            # Naive copy-paste for comparision
            mask3 = np.reshape(mask, [mask.shape[0], mask.shape[1], 1])
            I_naive = fg * mask3 + bg * (1 - mask3)

            utils.save_im(
                results_dir,
                [im_no_extensions[0], im_no_extensions[1], "gd", "copy", "paste"],
                I_dest,
            )
            utils.save_im(
                results_dir,
                [im_no_extensions[0], im_no_extensions[1], "naive", "copy", "paste"],
                I_naive,
            )

            if show_plots:
                plt.figure(figsize=(9, 9))

                plt.subplot(121)
                plt.title("Naive")
                plt.axis("off")
                plt.imshow(I_naive)

                plt.subplot(122)
                plt.title("Poisson Blending")
                plt.axis("off")
                plt.imshow(I_dest)

                plt.show()

        except Exception as e:
            if debug:
                raise e
            print(f"Error processing image {im_names}: {e}")
            continue
