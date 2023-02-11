import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.signal import convolve2d
from time import time
import pickle

from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D

np.random.seed(500)

N, D, K = 10, 3, 3

kernel = np.ones(shape=(D, D))

cache_path = 'ws_likelihood_approx_graphs.pickle'
cache_file = open(cache_path, 'rb')
CACHED = pickle.load(cache_file)


def is_valid_img(im, d, k):
    conv = convolve2d(im, kernel, mode='same')
    return np.sum(im[:-d, :-d]) == k and np.all(conv <= 1), conv


def is_well_separated(im, d):
    return np.all(np.convolve(np.sum(im, axis=1), np.ones(d), mode='valid') <= 1)


def calculate_totals(n, d, k):
    start = time()

    try:
        total_vws_valid, total_valid, total_combinations = CACHED[(n, d, k)]
    except:

        total_valid = 0
        total_vws_valid = 0
        total_combinations = 0
        for ii in itertools.product(*[np.arange(n ** 2) for _ in range(K)]):
            total_combinations += 1
            img = np.zeros((n * n))
            img[list(ii)] = 1
            img = img.reshape((n, n))
            is_valid, conv_img = is_valid_img(img, d=d, k=k)
            if is_valid:
                total_valid += 1

                if is_well_separated(img, d=d):
                    total_vws_valid += 1

        CACHED[(n, d, k)] = [total_vws_valid, total_valid, total_combinations]
        pickle.dump(CACHED, open(cache_path, 'wb'))

    print(f'({n}) Totals: {total_vws_valid} / {total_valid} / {total_combinations}, '
          f'took {time() - start} seconds')

    return total_vws_valid, total_valid, total_combinations


def __main__():
    all_total_valid = []
    all_total_vws_valid = []
    all_total_combinations = []
    ns = np.arange(8, 18, 1)
    print(f'For N in {ns}')
    for n in ns:
        total_vws_valid, total_valid, total_combinations = calculate_totals(n, D, K)

        all_total_valid.append(total_valid)
        all_total_vws_valid.append(total_vws_valid)
        all_total_combinations.append(total_combinations)

    all_total_valid = np.array(all_total_valid)
    all_total_vws_valid = np.array(all_total_vws_valid)
    all_total_combinations = np.array(all_total_combinations)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Total Valid  VWS Counts vs Total Valid Counts')
    ax[0].plot(ns, all_total_valid, 'o-', label='Total Valid Combinations')
    ax[0].plot(ns, all_total_vws_valid, 'o-', label='Total Valid VWS Combinations')
    ax[0].legend(loc="upper left")

    ax[1].set_title('Percentage of total valid VWS')
    ax[1].plot(ns, all_total_vws_valid / all_total_valid, 'o-')

    plt.show()


if __name__ == '__main__':
    __main__()
