import gen_mandelbrot as gm
import sampling_alg as sa

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
import pickle
from typing import Optional
from decimal import Decimal as dec

def calculate_area(func, bounds, s, i, antithetic, seed1=None, seed2=None, arr_samples:Optional[list]=[]):
    """
    Calculates the area of the mandelbrot set based on a sampling function.
    :param func: the sampling function
    :param bounds: real min,max ; imaginary min,max
    :param s: nr samples
    :param i: nr iterations
    :param antithetic: bool: if antithetic sample only
    :param seed1: seed 1 for the sampling method
    :param seed2: seed 2 for the sampling method
    :param arr_samples
    :return: the area of the mandelbrot set
    """
    # Initialize the grid and create samples
    rng = np.random.default_rng(seed1)
    rng2 = np.random.default_rng(seed2)
    re_min, re_max, im_min, im_max = bounds
    area_total = (np.abs(re_min) + np.abs(re_max)) * (np.abs(im_min) + np.abs(im_max))
    if len(arr_samples)==0:
        samples = func(re_min, re_max, im_min, im_max, rng, s, rng2, antithetic=antithetic)
    else:
        samples = arr_samples
    # Check if the samples are part of the MB-set
    res = gm.mandelbrot(samples, i)
    ct_res = Counter(res)

    # Calculate the area
    area_mandel = area_total * (ct_res[i - 1] / (sum(ct_res.values())))
    return area_mandel


def mc_area(bounds, samples, iter, N, anti):
    """
    Calculate the area for 3 sampling methods, for different nr samples and iterations
    :param bounds:  real min,max ; imaginary min,max
    :param samples: np array of nr samples to be tested e.g., [4,16,25]
    :param iter: np array of the nr iterations to be tested
    :param N: nr of reruns to be used to compute the average
    :param anti: if should use only inverted points from antithetic should be computed.
     Note, to compute the area via antithetic sampling, must use both non-antithetic and antithetic values.
    :return: np array of shape (#simulation, #iterations, #samples)
    """
    rng = np.random.default_rng(0)
    A_pr, A_lh, A_ot = [], [], []
    for i in iter:
        i_pr, i_lh, i_ot = [], [], []
        for s in samples:
            s_pr, s_lh, s_ot = [], [], []
            for n in range(N):
                if anti:
                    seed1 = s * n + rng.integers(0, 1000)
                    seed2 = s * (N + n) + rng.integers(0, 1000)

                    normal_pr = [calculate_area(sa.sample_pr, bounds, s, i, False, seed1=seed1, seed2=seed2)]
                    normal_lh = [calculate_area(sa.sample_lh, bounds, s, i, False, seed1=seed1, seed2=seed2)]
                    normal_ot = [calculate_area(sa.sample_ot, bounds, s, i, False, seed1=seed1, seed2=seed2)]

                    anti_pr = [calculate_area(sa.sample_pr, bounds, s, i, True, seed1=seed1, seed2=seed2)]
                    anti_lh = [calculate_area(sa.sample_lh, bounds, s, i, True, seed1=seed1, seed2=seed2)]
                    anti_ot = [calculate_area(sa.sample_ot, bounds, s, i, True, seed1=seed1, seed2=seed2)]

                    s_pr += [np.mean(np.array([normal_pr, anti_pr]), axis=1)]
                    s_lh += [np.mean(np.array([normal_lh, anti_lh]), axis=1)]
                    s_ot += [np.mean(np.array([normal_ot, anti_ot]), axis=1)]
                else:
                    s_pr += [[calculate_area(sa.sample_pr, bounds, s, i, antithetic=anti)]]
                    s_lh += [[calculate_area(sa.sample_lh, bounds, s, i, antithetic=anti)]]
                    s_ot += [[calculate_area(sa.sample_ot, bounds, s, i, antithetic=anti)]]
                print(f'Simulation {n + 1, i, s} done')
            i_pr += [[np.mean(s_pr), np.std(s_pr)]]
            i_lh += [[np.mean(s_lh), np.std(s_lh)]]
            i_ot += [[np.mean(s_ot), np.std(s_ot)]]
            # print(f"Sample {s} done", end='\r', flush=True)
        A_pr += [i_pr]
        A_lh += [i_lh]
        A_ot += [i_ot]
        print(f"Iteration {i} done")
    return np.array(A_pr), np.array(A_lh), np.array(A_ot)


def confidint(sigma, n, z):
    """
    Return confidence interval for random variable with n samples, std sigma, 
    and confidence level z
    :param sigma: Standard deviation of RV
    :param n: sample size
    :param z: confidence level
    :return: confidence interval
    """
    return (sigma * z) / np.sqrt(n)


def picklesave(A_pr, A_lh, A_ot, name):
    """
    save the results from mc_area to a pickle file
    :param A_pr: area for pure random sampling
    :param A_lh: area for latin hypercube
    :param A_ot: area for orthogonal
    :return: void
    """
    file_pr = open(f'pickle/area_pr_{name}', 'wb')
    pickle.dump(A_pr, file_pr)
    file_pr.close()
    file_lh = open(f'pickle/area_lh_{name}', 'wb')
    pickle.dump(A_lh, file_lh)
    file_lh.close()
    file_ot = open(f'pickle/area_ot_{name}', 'wb')
    pickle.dump(A_ot, file_ot)
    file_ot.close()
    return


def pickleopen(dir='', name=''):
    """
    open the results from mc_area
    :param dir:
    :return: A_pr, A_lh, A_ot / input to picklesave()
    """
    file_pr = open(f'{dir}area_pr{name}', 'rb')
    A_pr = pickle.load(file_pr)
    file_pr.close()
    file_lh = open(f'{dir}area_lh{name}', 'rb')
    A_lh = pickle.load(file_lh)
    file_lh.close()
    file_ot = open(f'{dir}area_ot{name}', 'rb')
    A_ot = pickle.load(file_ot)
    file_ot.close()
    return A_pr, A_lh, A_ot


def plotarea3D(A, samples, iterations):
    """
    Generate a 3d plot of the sample x iteration landscape.
    :param A: area values
    :param samples:  sample values
    :param iterations: iteration values
    :return: void
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(range(len(samples)), range(len(iterations)))
    ax.plot_surface(X, Y, A[:, :, 0], cmap=cm.coolwarm)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Iterations')
    ax.set_zlabel('Area')
    plt.xticks(range(len(samples)), samples)
    plt.yticks(range(len(iterations)), iterations)
    plt.show()
    return


def heatmap_se(data, samples, iterations):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='plasma')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(samples)), labels=samples)
    ax.set_yticks(np.arange(len(iterations)), labels=iterations)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Iterations')
    for i in range(len(iterations)):
        for j in range(len(samples)):
            text = ax.text(j, i, '%.2E' % dec(data[i, j]),
                        ha="center", va="center", color="w")

    plt.show()


def plotconv(A_pr, A_lh, A_ot, X, xax):
    """
    Plot the convergence with error bars of different sampling methods over a range of iterations
    :param A_pr: area for pure random sampling: np array
    :param A_lh: area for latin hypercube: np array
    :param A_ot: area for orthogonal : np array
    :param X: x axis values/ iterations
    :return: void
    """
    fig, ax = plt.subplots()
    plt.errorbar(  X, A_pr[:, 0], A_pr[:, 1], 
                                fmt='o-', color='blue', alpha=0.4,
                                capsize=5, elinewidth=1)
    _, _, err = plt.errorbar(   X, A_lh[:, 0], A_lh[:, 1], 
                                fmt='s--', color='red', alpha=0.4, 
                                capsize=5, elinewidth=1, linewidth=2)
    for e in err: e.set_linestyle('--'), e.set_linewidth(2)
    _, _, err = plt.errorbar(  X, A_ot[:, 0], A_ot[:, 1], 
                                fmt='x:', color='black', 
                                capsize=5, elinewidth=1)
    for e in err: e.set_linestyle(':'), e.set_linewidth(2)
    plt.plot(X, [1.50659177 for _ in range(len(X))], '--', color='grey', alpha=0.4)

    plt.legend(['Reference area', 'PR', 'LH', 'OT'])
    ax.set_xscale('log')
    plt.ylabel('Area')
    plt.xlabel(f'no. of {xax}')
    plt.grid()
    plt.show()
    return


if __name__ == "__main__":
    # parameters
    samples = [ i ** 2 for i in [10, 20, 32, 45, 64, 90, 100]]
    iterations = [100, 250, 500, 1000, 1500, 2000, 3000, 4000]
    bounds = -2, 0.47, -1.12, 1.12  # real-min,max,im-min,max

    # A_pr, A_lh, A_ot = mc_area(bounds, samples, iterations, 100, True)  # 100
    # picklesave(A_pr, A_lh, A_ot, '_anti100')
    A_pr, A_lh, A_ot = pickleopen('pickle/', '__anti100')

    heatmap_se(A_pr[:,:,1], samples, iterations)
    heatmap_se(A_lh[:,:,1], samples, iterations)
    heatmap_se(A_ot[:,:,1], samples, iterations)

    # ci_pr = confidint(A_pr[:,:,1], 100, 1.96)
    # ci_lh = confidint(A_lh[:,:,1], 100, 1.96)
    # ci_ot = confidint(A_ot[:,:,1], 100, 1.96)

    # print(A_pr, A_lh, A_ot)

    # Remove the first element of samples and iterations for a more clear plot
    # pr_it, lh_it, ot_it = A_pr[:,-1,:], A_lh[:,-1,:], A_ot[:,-1,:]
    # pr_sm, lh_sm, ot_sm = A_pr[-1,:,:], A_lh[-1,:,:], A_ot[-1,:,:]
    ## Shape of A:  A[iterations, samples, (area, ci)]
    # plotconv(pr_it[1:], lh_it[1:], ot_it[1:], iterations[1:], 'iterations')
    # plotconv(pr_sm[1:], lh_sm[1:], ot_sm[1:], samples[1:], 'samples')

    # plotarea3D(A_pr, samples, iterations)
    # plotarea3D(A_lh, samples, iterations)
    # plotarea3D(A_ot, samples, iterations)
