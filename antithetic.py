import sampling_alg as sa
import area as a
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt




def compare_antithetic_area(n_simulations,n_samples,iterations,bounds,rng, rng2, funcs=[sa.sample_pr,sa.sample_lh,sa.sample_ot]):
    """
    Generate area values for all sampling methods and their antithetic complements for later comparison.
    Note, here a single quantity of samples and iterations is expected, not an array.
    :param n_simulations: Nr of simulations
    :param n_samples: Nr of samples
    :param iterations: Nr of iterations
    :param bounds: real min, real max, imaginary min, imaginary max
    :param funcs: three sampling function to test
    :return: 2-tuple of np arrays of shape (n_simulations, 3) each: areas_normal, areas_anti
    """


    areas_anti = np.empty((n_simulations,3)) # simulation, sampling method
    areas_normal = np.empty((n_simulations,3))
    re_min, re_max, im_min, im_max = bounds
    for func_idx, samping_func in enumerate(funcs):
        for i in range(n_simulations):
            # run simulation normal
            arr_samples = samping_func(re_min, re_max, im_min, im_max, rng, n_samples, rng2, antithetic=False)
            area_1 = a.calculate_area(samping_func, bounds, s =n_samples, i=iterations,seed1=n_simulations,seed2=n_simulations+1,arr_samples=arr_samples,antithetic=False)
            # run simulation inverted
            arr_samples_anti = sa.convert_antithetic(arr_samples)
            area_2 = a.calculate_area(samping_func, bounds, s=n_samples, i=iterations,seed1=n_simulations*2,seed2=n_simulations*2+1,arr_samples=arr_samples_anti,antithetic=True)
            # take average area
            areas_anti[i,func_idx] = (area_1 + area_2)/2 # change 1 to sampling method
            areas_normal[i,func_idx] = area_1 # change 1 to sampling method
        print(f'{func_idx+1}/3 Sampling methods is done.')
    return areas_normal,areas_anti
# T-test on differences in Variances between each method and its antithetic complement.

def bootstrapped_t_test(normal, antithetic_variate, seed, n_resamples, statistic=np.var):
    """
    Compute bootstrapped t-test for any statistic/parameter between a sampling method and its antithetic complement.
    Or generally two numpy arrays.
    :param normal: area values from the non-antithetic sampling method
    :param antithetic_variate:  area values from the antithetic sampling method
    :param seed: random state for bootstrapping
    :param n_resamples: number of resamples; size of sample is len(values)
    :param statistic: parameter to test: e.g., np.mean, np.var, np.std...
    :return: independent ttest results
    """
    rng = np.random.default_rng(seed)
    boot_areas_normal = rng.choice(normal, size=len(normal) * n_resamples, replace=True).reshape(
        (len(normal), n_resamples))
    boot_variances_normal = statistic(boot_areas_normal, axis=0)

    boot_areas_anti = rng.choice(antithetic_variate, size=len(antithetic_variate) * n_resamples, replace=True).reshape(
        (len(antithetic_variate), n_resamples))
    boot_variances_anti = statistic(boot_areas_anti, axis=0)

    return ttest_ind(boot_variances_normal, boot_variances_anti)

def draw_bar_antithetic(areas_normal,areas_anti,boot_ci_normal,boot_ci_anti,fname="Barchart_anti"):
    plt.figure(figsize=(12,8))
    anti_var = np.var(areas_anti,axis=0)
    norm_var = np.var(areas_normal,axis=0)
    labels = ["Pure Random", "Latin Hypercube", "Orthogonal"]
    plt.bar(labels,norm_var,color='grey',yerr=boot_ci_normal.standard_error,capsize=10,label="non-antithetic")
    plt.bar(labels,anti_var,color='royalblue',yerr=boot_ci_anti.standard_error,edgecolor="None",linewidth=2,alpha=0.5,label="antithetic",capsize=10)
    plt.xlabel("Sampling method")
    plt.ylabel(fr"$Var(X)$")
    plt.legend()
    plt.savefig('figures/'+fname+'.svg', bbox_inches="tight")

# def draw_bar_antithetic(areas_normal, areas_anti, boot_ci_normal, boot_ci_anti, fname="Barchart_anti"):
#     """
#     Draw the barchart to compare effectiveness of variance reduction.
#     :param areas_normal: areas
#     :param areas_anti: antithetic areas
#     :param boot_ci_normal: output of scipy.stats.bootstrap for area
#     :param boot_ci_anti: output of scipy.stats.bootstrap for area (antithetic)
#     :param fname: filename
#     :return: void
#     """
#     plt.figure(figsize=(12, 8))
#     anti_var = np.var(areas_anti, axis=0)
#     norm_var = np.var(areas_normal, axis=0)
#     labels = ["Pure Random", "Latin Hypercube", "Orthogonal"]
#     plt.bar(labels, norm_var, color='grey', yerr=boot_ci_normal.standard_error, capsize=10)
#     plt.bar(labels, anti_var, color='royalblue', yerr=boot_ci_anti.standard_error, edgecolor="None", linewidth=2,
#             alpha=0.5, label="antithetic", capsize=10)
#     plt.xlabel("Sampling method")
#     plt.ylabel(fr"$Var(X)$")
#     plt.legend()
#     plt.savefig('figures/' + fname + '.svg', bbox_inches="tight")
#     return
