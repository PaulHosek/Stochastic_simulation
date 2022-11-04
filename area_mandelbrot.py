import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


cur_center = -0.8 + 0.0j
cur_extent = 3.0 + 3.0j

resulution = 256*2^20
# XY-quadrant that will be inspected
cur_width = resulution  # (x axis); real numbers
cur_height = resulution# (y axis); imaginary

cur_max_iter = resulution # convergence test search depth
#%%
@nb.vectorize
def mandelbrot(c, max_iter=cur_max_iter):
    """
    Convergence test by iteration for single complex number.
    """
    z = 0

    # test for divergence with finite iterations
    for k in range(max_iter):
        z = z**2 + c
        if np.absolute(z) > 2.0:  # if true then complex number diverges and is not part of set
            break

    return k

def generate_complex_grid(width=cur_width, height=cur_height, center=cur_center, extent=cur_extent):
    """
    Generate complex number grid to pass to the iteration function.
    """
    scale = max(extent.real / width, extent.imag / height)
    real_index_grid, imag_index_grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
    c_grid = center + (real_index_grid - width // 2 + (imag_index_grid - height //2) * 1j) * scale
    return c_grid


def compute_mandelbrot(width=cur_width, height=cur_height, center=cur_center, extent=cur_extent, max_iter=cur_max_iter):
    """
    Compute mandelbrot set by generating complex grid and testing for divergence for each point.
    """
    niters = np.zeros((width, height), int)
    scale = max(extent.real / width, extent.imag / height)

    c_grid = generate_complex_grid(width=cur_width, height=cur_height, center=cur_center, extent=cur_extent)

    return mandelbrot(c_grid, max_iter)




def random_samples(rng, boundaries, n):
    re_min, re_max, im_min, im_max = boundaries
    re = rng.uniform(low=re_min,high=re_max, size=n)
    im = rng.uniform(low=im_min,high=im_max, size=n) * 1j

    return re + im