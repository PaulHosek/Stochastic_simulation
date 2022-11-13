import numpy as np
from itertools import product
import math
rng = np.random.default_rng(0)

def sample_pr(re_min, re_max, im_min, im_max, rng, n_samples, rng2,antithetic=False):
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
    re = rng.uniform(low=re_min, high=re_max, size=n_samples)
    im = rng.uniform(low=im_min, high=im_max, size=n_samples) * 1j
    if antithetic:
        return -0.5-(re+0.5) + (-1*im)
    return re + im

def sample_lh(re_min, re_max, im_min, im_max, rng, n_samples, rng2,antithetic=False):
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

    im_samples = np.empty(n_samples)
    real_samples = np.empty(n_samples)

    # for each square, generate a uniform sample
    for square in range(n_samples):
        real_samples[square] = rng.uniform(low=real_grid[square], high=real_grid[square + 1])
        im_samples[square] = rng.uniform(low=im_grid[square], high=im_grid[square + 1])

    if antithetic:
        return -0.5 - (real_samples + 0.5) + (-1 * rng.permutation(im_samples * 1j))
    return real_samples + rng.permutation(im_samples * 1j)

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
    if n_samples != math.isqrt(n_samples)**2:
        raise ValueError("n_samples must be a power of 2.")
    n_subspaces = math.isqrt(n_samples)

    # create 2d square array with n_subspaces **2 entries and n_subspaces rows
    coordinates_1d = np.arange(n_subspaces ** 2)

    # randomly arrange subspaces
    rng_1.shuffle(coordinates_1d.reshape([n_subspaces, n_subspaces]))
    # randomly move points within each subspace
    imag = coordinates_1d + rng_1.random(size=n_subspaces ** 2)

    rng_2.shuffle(coordinates_1d.reshape([n_subspaces, n_subspaces]))

    # want every nth element of every kth subarray i.e., all first elems
    trans_coord = coordinates_1d.reshape([n_subspaces, n_subspaces]).T.reshape(-1)

    real = trans_coord.T + rng_2.random(size=n_subspaces ** 2)

    # 1. stretch/shrink the square to fit the length of the axis
    # 2. move square to minimal value of each axis
    imag *= ((im_max - im_min) / n_subspaces**2)
    imag += im_min
    real *= (re_max - re_min) / n_subspaces**2
    real += re_min
    if antithetic:
        return -0.5 - (real + 0.5) + (-1 * rng.permutation(imag * 1j))
    return real + imag * 1j