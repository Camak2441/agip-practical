import os.path as path
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json

plt.title("Gradient Response")
plt.xlabel("Original Gradient Magnitude")
plt.ylabel("Enhanced Gradient Magnitude")

points = np.array([[0, 0], [0.1, 0.3], [1, 1]])
xs, ys = np.unstack(points, axis=1)
plt.plot(xs, ys, marker="", linestyle="-")

plt.savefig(path.join("results5", "lin1.png"))
plt.show()


plt.title("Gradient Response")
plt.xlabel("Original Gradient Magnitude")
plt.ylabel("Enhanced Gradient Magnitude")

points = np.array([[0, 0], [0.02, 0], [0.0201, 0.0201], [1, 1]])
xs, ys = np.unstack(points, axis=1)
plt.plot(xs, ys, marker="", linestyle="-")

plt.savefig(path.join("results5", "denoise2.png"))
plt.show()


plt.title("Gradient Response")
plt.xlabel("Original Gradient Magnitude")
plt.ylabel("Enhanced Gradient Magnitude")

points = np.array([[0, 0], [0.004, 0], [0.00401, 0.00401], [1, 1]])
xs, ys = np.unstack(points, axis=1)
plt.plot(xs, ys, marker="", linestyle="-")

plt.savefig(path.join("results5", "denoise3.png"))
plt.show()
