import numpy as np
import numba as nb
import time
from enum import Enum
from src.utils.logsumexp import logsumexp
from scipy.signal import convolve

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

        self.log_prob_all_noise = self._calc_log_prob_all_is_noise()
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

    def _calc_log_prob_all_is_noise(self):
        y = self._data
        n = y.shape[0]
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        return - n * np.log(self._noise_std * (2 * np.pi) ** 0.5) + minus_1_over_twice_variance * np.sum(np.square(y))

    @staticmethod
    @nb.jit
    def _compute_log_size_s_very_well_separated_single_axis(n, k, d):
        """
        Compute log(1/|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping.
        """
        max_k_in_row = min(n // d, k)
        log_size_S_per_row_per_k = np.zeros(max_k_in_row)
        for k_in_row in range(1, max_k_in_row + 1):
            log_size_S_per_row_per_k[k_in_row - 1] = LengthEstimator2DSeparationMethod._compute_log_size_S_1d(n,
                                                                                                              k_in_row,
                                                                                                              d)

        mapping = np.full((n + 1, k + 1), -np.inf)
        mapping[:, 0] = 0
        for curr_k in range(1, k + 1):
            max_k = min(curr_k, max_k_in_row)
            for i in range(d, n + 1):
                mapping[i, curr_k] = logsumexp(
                    mapping[i - d, range(curr_k - 1, curr_k - max_k - 1, -1)] + log_size_S_per_row_per_k[:max_k])
                mapping[i, curr_k] = np.logaddexp(mapping[i, curr_k], mapping[i - 1, curr_k])
        return mapping[n, k]

    @staticmethod
    def _compute_log_pd(n, k, d):
        size_S = LengthEstimator2DSeparationMethod._compute_log_size_s_very_well_separated_single_axis(n, k, d)
        size_S_both_axis = size_S + np.log(2)  # Twice the size of one axis
        # TODO: size_S_both_axis -= size_intersection
        return -size_S_both_axis

    def _compute_log_very_well_separated_in_both_axis(self, n, k, d):
        pass

    @staticmethod
    def _compute_log_size_S_1d(n, k, d):
        """
        Compute log(1/|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping.
        """
        if k * d > n:
            return -np.inf
        n_tag = n - (d - 1) * k
        k_tag = k
        return log_binomial(n_tag, k_tag)

    def _calc_sum_yx_minus_x_squared(self, y, x):
        x_tag = np.flip(x)  # Flipping to cross-correlate
        sum_yx_minus_x_squared = convolve(-2 * y, x_tag, mode='valid')  # -2y to save computations later
        sum_yx_minus_x_squared += np.sum(np.square(x))
        sum_yx_minus_x_squared *= - 0.5 / self._noise_std ** 2
        return sum_yx_minus_x_squared

    @nb.jit
    def _calc_length_likelihood(self, signal_filter, expected_num_of_occurrences):
        y = self._data
        n = y.shape[0]
        d = signal_filter.shape[0]

        max_k_in_row = min(n // d, expected_num_of_occurrences)

        # Axis 1
        # per viable pixel compute the prob the filter is there (Can be done faster using convolution)
        sum_yx_minus_x_squared = self._calc_sum_yx_minus_x_squared(y, signal_filter)

        # Per row compute the sum over all s_k
        pre_compute_per_row_per_k = np.zeros((n + 1, max_k_in_row))
        for j in range(n + 1 - d):
            # Go to 1d case
            curr_sum_yx_minus_x_squared = sum_yx_minus_x_squared[j]
            mapping = np.full((n + 1, max_k_in_row + 1), -np.inf)
            mapping[:, 0] = 0
            for curr_k in range(1, max_k_in_row + 1):
                for i in range(n - d, -1, -1):
                    mapping[i, curr_k] = np.logaddexp(curr_sum_yx_minus_x_squared[i] + mapping[i + d, curr_k - 1],
                                                      mapping[i + 1, curr_k])
            pre_compute_per_row_per_k[j] = mapping[0, 1:]

        # Allocating memory
        # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
        # when k=0 the probability is 1
        mapping = np.full((n + 1, expected_num_of_occurrences + 1), -np.inf)
        mapping[:, 0] = 0

        # Filling values one by one, skipping irrelevant values
        mapping = np.full((n + 1, expected_num_of_occurrences + 1), -np.inf)
        mapping[:, 0] = 0
        for curr_k in range(1, expected_num_of_occurrences + 1):
            max_k = min(curr_k, max_k_in_row)
            for i in range(d, n + 1):
                mapping[i, curr_k] = logsumexp(
                    mapping[i - d, range(curr_k - 1, curr_k - max_k - 1, -1)] + pre_compute_per_row_per_k[i, :max_k])
                mapping[i, curr_k] = np.logaddexp(mapping[i, curr_k], mapping[i - 1, curr_k])
        very_well_separated_axis_1 = mapping[n, expected_num_of_occurrences]

        # Axis 2
        # Doing the same for y transpose to find very well separated on the other axis
        sum_yx_minus_x_squared = sum_yx_minus_x_squared.T.copy()
        pre_compute_per_row_per_k = np.zeros((n + 1, max_k_in_row))
        for j in range(n + 1 - d):
            # Go to 1d case
            curr_sum_yx_minus_x_squared = sum_yx_minus_x_squared[j]
            mapping = np.full((n + 1, max_k_in_row + 1), -np.inf)
            mapping[:, 0] = 0
            for curr_k in range(1, max_k_in_row + 1):
                for i in range(n - d, -1, -1):
                    mapping[i, curr_k] = np.logaddexp(curr_sum_yx_minus_x_squared[i] + mapping[i + d, curr_k - 1],
                                                      mapping[i + 1, curr_k])
            pre_compute_per_row_per_k[j] = mapping[0, 1:]

        mapping = np.full((n + 1, expected_num_of_occurrences + 1), -np.inf)
        mapping[:, 0] = 0
        for curr_k in range(1, expected_num_of_occurrences + 1):
            max_k = min(curr_k, max_k_in_row)
            for i in range(d, n + 1):
                mapping[i, curr_k] = logsumexp(
                    mapping[i - d, range(curr_k - 1, curr_k - max_k - 1, -1)] + pre_compute_per_row_per_k[i, :max_k])
                mapping[i, curr_k] = np.logaddexp(mapping[i, curr_k], mapping[i - 1, curr_k])
        very_well_separated_axis_2 = mapping[n, expected_num_of_occurrences]

        very_well_separated_both_axis = np.logaddexp(very_well_separated_axis_1, very_well_separated_axis_2)

        # TODO: remove intersection

        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, expected_num_of_occurrences, d)
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
