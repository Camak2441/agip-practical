import os.path as path
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json

dimensions = []
sparse_solver_times = []
cholesky_times = []

timings = {}

with open(path.join("results5", "timings.json")) as file:
    timings = json.load(file)

for run in timings:
    run_data = timings[run]
    dimensions.append([run_data["width"], run_data["height"]])
    sparse_solver_times.append(run_data["sparse_solver_time"])
    cholesky_times.append(run_data["cholesky_time"])

dimensions = np.array(dimensions)
sparse_solver_times = np.array(sparse_solver_times)
cholesky_times = np.array(cholesky_times)

widths, heights = np.unstack(dimensions, axis=1)

plt.title("Reconstruction Time")
plt.xlabel("Image Size (pixels)")
plt.ylabel("Time (seconds)")

plt.plot(widths * heights, cholesky_times, marker="x", linestyle="", label="chlmod.cholesky")
plt.plot(widths * heights, sparse_solver_times, marker="x", linestyle="", label="scipy.spsolve")
plt.figlegend()

plt.savefig(path.join("results5", "timings.png"))
plt.show()
