import numpy as np
import numba as nb
import time
from enum import Enum
from src.utils.logsumexp import logsumexp
from scipy.signal import convolve
import src.algorithms.utils as utils

from src.algorithms.utils import log_binomial
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator as SPE
from src.utils.logger import logger


class SignalPowerEstimator(SPE, Enum):
    Exact = "Exact Power"


class LengthEstimator2DSeparationMethod:

    def __init__(self,
                 data,
                 length_options,
                 signal_area_fraction_boundaries,
                 signal_num_of_occurrences_boundaries,
                 signal_filter_gen,
                 noise_mean,
                 noise_std,
                 signal_power_estimator_method,
                 exp_attr,
                 num_of_power_options=10,
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._signal_area_fraction_boundaries = signal_area_fraction_boundaries
        self._signal_num_of_occurrences_boundaries = signal_num_of_occurrences_boundaries
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._exp_attr = exp_attr
        self._num_of_power_options = num_of_power_options

        self._n = self._data.shape[0]

        self.log_prob_all_noise = utils.log_prob_all_is_noise(self._data, self._noise_std)
        self._signal_power = self._estimate_full_signal_power()
        logger.debug(f'Full signal power: {self._signal_power}')

        self._signal_instance_power_options_estimation = self._estimate_possible_instance_of_signal_power()
        logger.info(f'Power options are: {self._signal_instance_power_options_estimation}')

        # self._signal_instance_power_options_estimation = [0.46, 0.52, 0.58]

    def _estimate_possible_instance_of_signal_power(self):
        self._signal_instance_power_options_estimation = np.zeros_like(self._num_of_power_options)
        f_min, f_max = self._signal_area_fraction_boundaries
        xxx = np.round(np.linspace(f_min, f_max, self._num_of_power_options), 2)

        p_max = np.round(self._signal_power / (np.prod(self._data.shape) * f_min), 2)
        p_min = np.round(self._signal_power / (np.prod(self._data.shape) * f_max), 2)

        # power_options = np.round(np.linspace(p_min, p_max, self._num_of_power_options), 2)
        power_options = np.round(self._signal_power / (np.prod(self._data.shape) * xxx), 2)

        return power_options

    def _estimate_full_signal_power(self):
        # If we know the exact signal power we use it, else compute from data
        if self._signal_power_estimator_method == SignalPowerEstimator.Exact:
            d, p, k = self._exp_attr["d"], self._exp_attr["p"], self._exp_attr["k"]
            signal_power = self._estimate_single_instance_of_signal_power(d, p * k)
        else:
            signal_power = estimate_signal_power(self._data, self._noise_std, self._noise_mean,
                                                 method=self._signal_power_estimator_method)
        return signal_power

    def _estimate_single_instance_of_signal_power(self, signal_length, signal_power):
        return np.sum(self._signal_filter_gen(signal_length, signal_power))

    def _estimate_num_of_signal_occurrences(self, signal_length, signal_power):
        single_signal_power = self._estimate_single_instance_of_signal_power(signal_length, signal_power)
        k = int(np.round(self._signal_power / single_signal_power))
        return k

    def _calc_length_likelihood(self, signal_filter, expected_num_of_occurrences):
        y = self._data
        n = y.shape[0]
        d = signal_filter.shape[0]

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

    def _calc_signal_length_likelihood(self, signal_length):
        likelihoods = np.zeros_like(self._signal_instance_power_options_estimation)
        min_occurrences, max_occurrences = self._signal_num_of_occurrences_boundaries

        tic = time.time()

        for i, signal_power in enumerate(self._signal_instance_power_options_estimation):
            expected_num_of_occurrences = self._estimate_num_of_signal_occurrences(signal_length, signal_power)

            single_signal_area = np.count_nonzero(self._signal_filter_gen(signal_length, signal_power))
            total_signal_area = single_signal_area * expected_num_of_occurrences
            signal_area_fraction = total_signal_area / np.prod(self._data.shape)
            logger.debug(f'Calculating likelihood for length={signal_length} and power={signal_power}.'
                  f'Expected occurrences is: {expected_num_of_occurrences},'
                  f'Total area fraction is {signal_area_fraction}')

            if expected_num_of_occurrences < min_occurrences or expected_num_of_occurrences > max_occurrences:
                likelihood = -np.inf
            else:
                signal_filter = self._signal_filter_gen(signal_length, signal_power)
                likelihood = self._calc_length_likelihood(signal_filter, expected_num_of_occurrences)
            logger.debug(f'Likelihood = {likelihood}\n')

            likelihoods[i] = likelihood

        toc = time.time()

        if self._logs:
            logger.debug(
                f"For Length={signal_length} took total time of {toc - tic} seconds")

        return likelihoods

    def estimate(self):
        likelihoods = np.array([self._calc_signal_length_likelihood(d) for d in self._length_options])

        likelihoods_dict = {}
        for i, signal_power in enumerate(self._signal_instance_power_options_estimation):
            likelihoods_dict[f'p_{signal_power}'] = likelihoods[:, i]

        # max_likely_length = self._length_options[np.argmax(likelihoods)]
        return likelihoods_dict, 0
