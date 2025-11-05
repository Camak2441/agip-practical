import time
import os.path as path
import sys
import os
import skimage.io as io
import skimage
import numpy as np
import matplotlib.pyplot as plt

from task4_grad_domain import img2grad_field, reconstruct_grad_field

import utils

scales = 11

runs = 5
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
            case "--scales" | "-l":
                index += 1
                if index < len(args):
                    scales = int(args[index])
                    index += 1
                continue
            case "--runs" | "-r":
                index += 1
                if index < len(args):
                    runs = int(args[index])
                    index += 1
                continue

        im_name = args[index]
        im_no_extension = utils.remove_file_extension(im_name)
        index += 1

        dimensions = []
        sparse_solver_times = []
        cholesky_times = []

        im = io.imread(path.join("images", im_name), as_gray=True)
        im = skimage.img_as_float(im)

        pyramid = [
            skimage.transform.rescale(im, np.sqrt(area_scale))
            for area_scale in np.linspace(0, 1, num=scales)
        ]

        for layer in pyramid:
            print("Timing reconstruction for layer of size", layer.shape)
            sparse_solver_samples = []
            cholesky_samples = []
            G = img2grad_field(layer)
            Gm = np.sqrt(np.sum(G * G, axis=2))

            w = 1 / (Gm + 0.0001)  # To avoid pinching artefacts
            for run in range(runs):
                print("Running scipy.spsolve run", run)
                start_time = time.time()
                imr = reconstruct_grad_field(
                    G, w, layer[0, 0], layer, cholesky_decomp=False
                )
                end_time = time.time()
                sparse_solver_samples.append(end_time - start_time)

            for run in range(runs):
                print("Running cholmod.cholesky run", run)
                start_time = time.time()
                imr = reconstruct_grad_field(
                    G, w, layer[0, 0], layer, cholesky_decomp=True
                )
                end_time = time.time()
                cholesky_samples.append(end_time - start_time)

            dimensions.append([layer.shape[1], layer.shape[0]])
            sparse_solver_times.append(sparse_solver_samples)
            cholesky_times.append(cholesky_samples)

        dimensions = np.array(dimensions)
        sparse_solver_times = np.array(sparse_solver_times)
        cholesky_times = np.array(cholesky_times)
        widths, heights = np.unstack(dimensions, axis=1)
        plt.title("Reconstruction Time")
        plt.xlabel("Image Size (pixels)")
        plt.ylabel("Time (seconds)")

        plt.plot(
            widths * heights,
            np.sum(cholesky_times, axis=1) / runs,
            marker="x",
            linestyle="solid",
            label="chlmod.cholesky",
        )
        plt.plot(
            widths * heights,
            np.sum(sparse_solver_times, axis=1) / runs,
            marker="x",
            linestyle="solid",
            label="scipy.spsolve",
        )
        plt.figlegend()

        plt.savefig(
            path.join(results_dir, im_no_extension + "_" + str(scales) + "_timings.png")
        )

        if show_plots:
            plt.show()
