import numpy as np
import time
from enum import Enum
import src.algorithms.utils as utils

from src.algorithms.signal_power_estimator import SignalPowerEstimator as SPE
from src.utils.logger import logger


class SignalPowerEstimator(SPE, Enum):
    Exact = "Exact Power"


class LengthEstimator2DSeparationMethod:

    def __init__(self,
                 data,
                 length_options,
                 power_options,
                 num_of_occ_estimation,
                 num_of_occ_estimation_mask,
                 signal_filter_gen,
                 noise_mean,
                 noise_std,
                 exp_attr,
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._power_options = power_options
        self._num_of_occ_estimation = num_of_occ_estimation
        self._num_of_occ_estimation_mask = num_of_occ_estimation_mask
        self._num_of_occ = 10  # Should get as input
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._logs = logs
        self._exp_attr = exp_attr

        self._n = self._data.shape[0]
        self.log_prob_all_noise = utils.log_prob_all_is_noise(self._data, self._noise_std)

    def _calc_signal_length_likelihood(self, length_idx):
        signal_length = self._length_options[length_idx]
        logger.info(f'Calculating likelihood for length={signal_length}')

        tic = time.time()
        likelihood, power = utils.max_argmax_2d_case(self._data, self._signal_filter_gen(signal_length, 1),
                                                     self._num_of_occ, self._noise_std)
        logger.debug(f'Likelihood = {likelihood}\n')
        toc = time.time()

        if self._logs:
            logger.debug(
                f"For Length={signal_length} took total time of {toc - tic} seconds")

        return likelihood

    def estimate(self):
        likelihoods = np.array([self._calc_signal_length_likelihood(i) for i in range(len(self._length_options))])

        likelihoods_dict = {}
        for i, signal_power in enumerate(self._power_options):
            likelihoods_dict[f'p_{signal_power}'] = likelihoods[:, i]

        likelihoods_dict['max'] = np.max(likelihoods, axis=1)
        max_likely_length = self._length_options[np.argmax(likelihoods_dict['max'])]
        return likelihoods_dict, max_likely_length
