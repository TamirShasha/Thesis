import numpy as np
import time
from enum import Enum
import matplotlib.pyplot as plt
import os

import src.algorithms.utils as utils
from src.algorithms.utils import calc_most_likelihood_and_optimized_power_2d, cryo_downsample
from src.algorithms.filter_estimator_2d import FilterEstimator2D, create_filter_basis
from src.algorithms.signal_power_estimator import SignalPowerEstimator as SPE
from src.utils.logger import logger


class SignalPowerEstimator(SPE, Enum):
    Exact = "Exact Power"


class LengthEstimator2DVeryWellSeparated:

    def __init__(self,
                 data,
                 length_options,
                 fixed_num_of_occurrences,
                 signal_filter_gen,
                 noise_mean,
                 noise_std,
                 downsample_to_num_of_rows=2000,
                 logs=True,
                 experiment_dir=None):
        self._data = data
        self._length_options = length_options
        self._fixed_num_of_occurrences = fixed_num_of_occurrences
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._logs = logs
        self._experiment_dir = experiment_dir

        self._n = self._data.shape[0]
        if self._n > downsample_to_num_of_rows:
            self._downsample_factor = (self._n / downsample_to_num_of_rows)
            self._data = cryo_downsample(self._data, (downsample_to_num_of_rows, downsample_to_num_of_rows))
            self._noise_std = self._noise_std / self._downsample_factor
            self._n = self._data.shape[0]
            # plt.imshow(self._data, cmap='gray')
            # plt.show()
            self._length_options = np.array(np.ceil(self._length_options / self._downsample_factor), dtype=int)
            logger.info(f'Length options after downsample: {self._length_options}')

        self.log_prob_all_noise = utils.log_prob_all_is_noise(self._data, self._noise_std)

    def estimate_2(self):
        likelihoods = np.zeros(len(self._length_options))
        powers = np.zeros(len(self._length_options))
        for i, length in enumerate(self._length_options):
            likelihoods[i], powers[i] = calc_most_likelihood_and_optimized_power_2d(self._data,
                                                                                    self._signal_filter_gen(length, 1),
                                                                                    self._fixed_num_of_occurrences,
                                                                                    self._noise_std)
            logger.info(
                f'For length {self._length_options[i]} matched power is {powers[i]}, Likelihood={likelihoods[i]}')

        most_likely_length = self._length_options[np.nanargmax(likelihoods)]
        most_likely_power = powers[np.nanargmax(likelihoods)]
        print(most_likely_power)
        return likelihoods, most_likely_length, most_likely_power

    def estimate(self):

        likelihoods = np.zeros(len(self._length_options))
        for i, length in enumerate(self._length_options):
            # filter_basis = create_basis(length, 5)
            filter_basis = create_filter_basis(length, 10)
            filter_estimator = FilterEstimator2D(self._data, filter_basis, 30, self._noise_std)

            likelihoods[i], optimal_coeffs = filter_estimator.estimate()
            est_signal = filter_basis.T.dot(optimal_coeffs)

            plt.imshow(est_signal)
            plt.colorbar()
            fig_path = os.path.join(self._experiment_dir, f'_{length}.png')
            plt.savefig(fname=fig_path)
            plt.close()

            logger.info(
                f'For length {self._length_options[i]}, Likelihood={likelihoods[i]}')

        return likelihoods, 0, 0
