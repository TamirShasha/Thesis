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
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._logs = logs
        self._exp_attr = exp_attr

        self._n = self._data.shape[0]
        self.log_prob_all_noise = utils.log_prob_all_is_noise(self._data, self._noise_std)

    def _calc_length_likelihood(self, signal_filter, expected_num_of_occurrences):
        y = self._data
        n = y.shape[0]
        d = signal_filter.shape[0]
        expected_num_of_occurrences = 1

        # Axis 1
        sum_yx_minus_x_squared = utils.log_probability_filter_on_each_pixel(y, signal_filter, self._noise_std)
        mapping = utils.dynamic_programming_2d(n, expected_num_of_occurrences, d, sum_yx_minus_x_squared)
        very_well_separated_axis_1 = mapping[0, expected_num_of_occurrences]

        # Axis 2
        sum_yx_minus_x_squared = sum_yx_minus_x_squared.T
        mapping = utils.dynamic_programming_2d(n, expected_num_of_occurrences, d, sum_yx_minus_x_squared)
        very_well_separated_axis_2 = mapping[0, expected_num_of_occurrences]

        very_well_separated_both_axis = np.logaddexp(very_well_separated_axis_1, very_well_separated_axis_2)

        # Computing remaining parts of log-likelihood
        log_pd = - (np.log(2) + utils.log_size_S_2d_1axis(n, expected_num_of_occurrences, d))
        log_prob_all_noise = self.log_prob_all_noise

        likelihood = log_pd + log_prob_all_noise + very_well_separated_both_axis
        return likelihood

    def _calc_signal_length_likelihood(self, length_idx):
        signal_length = self._length_options[length_idx]
        likelihoods = np.zeros_like(self._power_options)
        possible_powers_mask = self._num_of_occ_estimation_mask[:, length_idx]
        expected_num_of_occurrences = self._num_of_occ_estimation[:, length_idx]

        logger.info(f'Calculating likelihood for length={signal_length}')

        tic = time.time()
        for i, signal_power in enumerate(self._power_options):

            if not possible_powers_mask[i]:
                likelihood = -np.inf
            else:
                signal_filter = self._signal_filter_gen(signal_length, signal_power)
                likelihood = self._calc_length_likelihood(signal_filter, expected_num_of_occurrences[i])
            logger.debug(f'Likelihood = {likelihood}\n')

            likelihoods[i] = likelihood

        toc = time.time()

        if self._logs:
            logger.debug(
                f"For Length={signal_length} took total time of {toc - tic} seconds")

        return likelihoods

    def estimate(self):
        likelihoods = np.array([self._calc_signal_length_likelihood(i) for i in range(len(self._length_options))])

        likelihoods_dict = {}
        for i, signal_power in enumerate(self._power_options):
            likelihoods_dict[f'p_{signal_power}'] = likelihoods[:, i]

        likelihoods_dict['max'] = np.max(likelihoods, axis=1)
        max_likely_length = self._length_options[np.argmax(likelihoods_dict['max'])]
        return likelihoods_dict, max_likely_length
