import numpy as np

rng = np.random.default_rng(0)


def sample_pr(re_min, re_max, im_min, im_max,rng, n_samples):
    """
    Pure random sampling
    :param re_min: minimal real value
    :param re_max: maximal real value
    :param im_min: minimal imaginary value
    :param im_max: maximal imaginary value
    :param n_samples: number of samples drawn
    :return: complex number array of len(n_samples)
    """
    print(re_min,re_max)
    re = rng.uniform(low=re_min, high=re_max, size=n_samples)
    im = rng.uniform(low=im_min, high=im_max, size=n_samples) * 1j
    return re + im


def sample_lh(re_min, re_max, im_min, im_max,rng, n_samples):
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

    # convert into complex array
    complex_2_col = np.vstack((rng.permutation(real_samples), im_samples * 1j)).T
    return complex_2_col[:, 0] + complex_2_col[:, 1]


re_min, re_max = -2, 0.47,
im_min, im_max = -1.12, 1.12
s = 10 ** 5 # sample size
i = 10**4   # iterations

samples = sample_pr(re_min,re_max,im_min,im_max,rng, s)
print(samples)