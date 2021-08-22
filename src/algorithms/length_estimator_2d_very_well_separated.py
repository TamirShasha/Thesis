import numpy as np
import time
from enum import Enum
import src.algorithms.utils as utils
from src.algorithms.utils import calc_most_likelihood_and_optimized_power_2d

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
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._fixed_num_of_occurrences = fixed_num_of_occurrences
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._logs = logs

        self._n = self._data.shape[0]
        self.log_prob_all_noise = utils.log_prob_all_is_noise(self._data, self._noise_std)

    def estimate(self):
        likelihoods = np.zeros(len(self._length_options))
        powers = np.zeros(len(self._length_options))
        for i, length in enumerate(self._length_options):
            likelihoods[i], powers[i] = calc_most_likelihood_and_optimized_power_2d(self._data,
                                                                                    self._signal_filter_gen(length, 1),
                                                                                    self._fixed_num_of_occurrences, self._noise_std)
            logger.info(
                f'For length {self._length_options[i]} matched power is {powers[i]}, Likelihood={likelihoods[i]}')

        most_likely_length = self._length_options[np.nanargmax(likelihoods)]
        most_likely_power = powers[np.nanargmax(likelihoods)]
        print(most_likely_power)
        return likelihoods, most_likely_length, most_likely_power
