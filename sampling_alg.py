import numpy as np
from itertools import product
import math

rng = np.random.default_rng(0)
from collections import Counter
import numba as nb


def sample_pr(re_min, re_max, im_min, im_max, rng, n_samples, rng2, antithetic=False):
    """
    Pure random sampling
    :param re_min: minimal real value
    :param re_max: maximal real value
    :param im_min: minimal imaginary value
    :param im_max: maximal imaginary value
    :param n_samples: number of samples drawn
    :return: complex number array of len(n_samples)
    """
    # print(re_min, re_max)
    if antithetic:
        n_samples //= 2
    re = rng.uniform(low=re_min, high=re_max, size=n_samples)
    im = rng.uniform(low=im_min, high=im_max, size=n_samples)

    if antithetic:
        # re = np.array_split(re, 2)[0]
        # im = np.array_split(im, 2)[0]

        re_anti = -1 * re - 1.54
        im_anti = -1 * im
        return re_anti + im_anti * 1j
        # return np.concatenate((re, re_anti)) + np.concatenate((im, im_anti)) * 1j
    return re + im * 1j


def sample_lh(re_min, re_max, im_min, im_max, rng, n_samples, rng2, antithetic=False):
    """
    Latin hypercube sampling.

    :param re_min: minimal real value
    :param re_max: maximal real value
    :param im_min: minimal imaginary value
    :param im_max: maximal imaginary value
    :param n_samples: number of samples drawn
    :param rng: np random generator object
    :return: complex number array of len(n_samples)
    """

    # generate 2d grid with equal spacing
    real_grid = np.linspace(re_min, re_max, n_samples + 1)
    im_grid = np.linspace(im_min, im_max, n_samples + 1)

    im = np.empty(n_samples)
    re = np.empty(n_samples)

    # for each square, generate a uniform sample
    for square in range(n_samples):
        re[square] = rng.uniform(low=real_grid[square], high=real_grid[square + 1])
        im[square] = rng.uniform(low=im_grid[square], high=im_grid[square + 1])

    if antithetic:
        rng.shuffle(im)

        re_anti = -1 * re - 1.54
        im_anti = -1 * im
        return re_anti + im_anti * 1j
    return re + rng.permutation(im * 1j)


def sample_ot(re_min, re_max, im_min, im_max, rng_1, n_samples, rng_2, antithetic=False):
    """
    Orthogonal sampling

    :param re_min: minimal real value
    :param re_max: maximal real value
    :param im_min: minimal imaginary value
    :param im_max: maximal imaginary value
    :param n_samples: number of samples drawn. Must be a power of 2/ have integer sqrt for subspace generation.
                        Number subspaces will be sqrt(n_samples)
    :param rng: np random generator object
    :return: complex number array of len(n_samples)
    """

    if n_samples != math.isqrt(n_samples) ** 2:
        raise ValueError("n_samples must be a power of 2.")

    n_subspaces = math.isqrt(n_samples)

    # create 2d square array with n_subspaces **2 entries and n_subspaces rows
    coordinates_1d = np.arange(n_subspaces ** 2)

    # randomly arrange subspaces
    rng_1.shuffle(coordinates_1d.reshape([n_subspaces, n_subspaces]))
    # randomly move points within each subspace
    im = coordinates_1d + rng_1.random(size=n_subspaces ** 2)

    rng_2.shuffle(coordinates_1d.reshape([n_subspaces, n_subspaces]))

    # want every nth element of every kth subarray i.e., all first elems
    trans_coord = coordinates_1d.reshape([n_subspaces, n_subspaces]).T.reshape(-1)

    re = trans_coord.T + rng_2.random(size=n_subspaces ** 2)

    # 1. stretch/shrink the square to fit the length of the axis
    # 2. move square to minimal value of each axis
    im *= ((im_max - im_min) / n_subspaces ** 2)
    im += im_min
    re *= (re_max - re_min) / n_subspaces ** 2
    re += re_min

    if antithetic:
        # re = np.array_split(re, 2)[0]
        # im = np.array_split(im, 2)[0]

        re_anti = -1 * re - 1.54
        im_anti = -1 * im
        return re_anti + im_anti * 1j
        # return np.concatenate((re, re_anti)) + np.concatenate((im, im_anti)) * 1j
    return re + im * 1j


def convert_antithetic(complex_points, ):
    """
    Converts a set of sample points into their antithetic/ inverted equivalent.
    Note: For an antithetic sampling, must use original and inverted points in separate simulations.
    """
    re_anti = -1 * complex_points.real - 1.54
    im_anti = -1 * complex_points.imag
    return re_anti + im_anti * 1j
