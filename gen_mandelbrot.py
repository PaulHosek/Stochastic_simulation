import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

########## DEFAULT VALUES #####################

cur_center = -0.8 + 0.0j
cur_extent = 3.0 + 3.0j
resolution = 256*4
cur_width = resolution  # (x axis); real numbers
cur_height = resolution  # (y axis); imaginary
cur_max_iter = resolution  # convergence test search depth
##############################################

@nb.vectorize(nopython=True)
def mandelbrot(c, max_iter=cur_max_iter):
    """
    Convergence test by iteration for single complex number.
    Takes in an array of complex numbers.
    Parallelises monte-carlo integration by circumventing python's GIL.
    """
    z = 0

    # test for divergence with finite iterations
    for k in range(max_iter):
        z = z ** 2 + c
        if np.absolute(z) > 2.0:  # if true then complex number diverges and is not part of set
            break

    return k


def generate_complex_grid(width=cur_width, height=cur_height, center=cur_center, extent=cur_extent):
    """
    Generate complex number grid to pass to the iteration function.
    This function is needed to draw the images.
    """
    scale = max(extent.real / width, extent.imag / height)
    real_index_grid, imag_index_grid = np.meshgrid(np.arange(0, width), np.arange(0, height))
    c_grid = center + (real_index_grid - width // 2 + (imag_index_grid - height // 2) * 1j) * scale
    return c_grid


def compute_mandelbrot(width=cur_width, height=cur_height, center=cur_center, extent=cur_extent, max_iter=cur_max_iter):
    """
    Compute mandelbrot set by generating complex grid and testing for divergence for each point.
    This function is needed to draw the images.
    """
    niters = np.zeros((width, height), int)
    scale = max(extent.real / width, extent.imag / height)

    c_grid = generate_complex_grid(width=width, height=height, center=center, extent=extent)

    return mandelbrot(c_grid, max_iter)

def draw_mandelbrot(cur_center, cur_extent,fname, resolution=256, color_list = ["black","white","darkorange","purple","black"]):
    cur_width = resolution  # (x axis); real numbers
    cur_height = resolution# (y axis); imaginary

    cur_max_iter = resolution # convergence test search depth

    niters = compute_mandelbrot(width=cur_width, height=cur_height, center=cur_center, extent=cur_extent, max_iter=cur_max_iter) # 17 ms
    fig, ax = plt.subplots(1,1, figsize=(12,12))
    c0, c1 = cur_center - cur_extent / 2, cur_center + cur_extent / 2
    plot_extent = (c0.real, c1.real, c0.imag, c1.imag)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_list) # cyan, lawngreen, white, magenta
    ax.imshow(niters, origin='lower', extent=plot_extent, cmap=cmap)
    ax.set_xlabel("$\Re(c)$")
    ax.set_ylabel("$\Im(c)$")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # plt.show()
    plt.savefig('figures/'+fname+'.svg', bbox_inches="tight")

