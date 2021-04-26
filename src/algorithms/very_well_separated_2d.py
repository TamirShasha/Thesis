import numpy as np
import numba as nb
import time
from enum import Enum
from src.utils.logsumexp import logsumexp
from scipy.signal import fftconvolve

from src.algorithms.utils import log_binomial
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator as SPE


class SignalPowerEstimator(SPE, Enum):
    Exact = "Exact Power"


class LengthExtractor2D:

    def __init__(self, y, length_options, signal_filter_gen,
                 noise_mean, noise_std, signal_power_estimator_method, exp_attr, logs=True):
        self._y = y
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._exp_attr = exp_attr
        self._n = self._y.shape[0]

        self.log_prob_all_noise = self._calc_log_prob_all_is_noise()
        self._signal_power = self._calc_signal_power(self._y)

    def _calc_signal_power(self, y):
        # If we know the exact signal power we use it, else compute from data
        if self._signal_power_estimator_method == SignalPowerEstimator.Exact:
            signal_power = self._calc_single_instance_of_signal_power(self._exp_attr["d"]) * self._exp_attr["k"]
        else:
            signal_power = estimate_signal_power(y, self._noise_std, self._noise_mean,
                                                 method=self._signal_power_estimator_method)
        return signal_power

    def _calc_single_instance_of_signal_power(self, d):
        return np.sum(np.power(self._signal_filter_gen(d), 2))

    def _find_expected_occurrences(self, y, d):
        single_signal_power = self._calc_single_instance_of_signal_power(d)
        k = int(np.round(self._signal_power / single_signal_power))
        return k

    def _calc_log_prob_all_is_noise(self):
        y = self._y
        n = y.shape[0]
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        return - n * np.log(self._noise_std * (2 * np.pi) ** 0.5) + minus_1_over_twice_variance * np.sum(np.square(y))

    @staticmethod
    @nb.jit
    def _compute_log_pd(n, k, d):
        """
        Compute log(1/|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping.
        """
        max_k_in_row = min(n // d, k)
        log_size_S_per_row_per_k = np.zeros(max_k_in_row)
        for k_in_row in range(1, max_k_in_row + 1):
            log_size_S_per_row_per_k[k_in_row - 1] = LengthExtractor2D._compute_log_size_S_1d(n, k_in_row, d)

        mapping = np.full((n + 1, k + 1), -np.inf)
        mapping[:, 0] = 0
        for curr_k in range(1, k + 1):
            max_k = min(curr_k, max_k_in_row)
            for i in range(d, n+1):
                mapping[i, curr_k] = logsumexp(mapping[i - d, range(curr_k-1, curr_k-max_k-1, -1)] + log_size_S_per_row_per_k[:max_k])
                mapping[i, curr_k] = np.logaddexp(mapping[i, curr_k], mapping[i - 1, curr_k])
        return -mapping[n, k]

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
        n = y.shape[0]
        d = x.shape[0]
        sum_yx_minus_x_squared = np.empty((n - d + 1, n - d + 1))
        z = fftconvolve(y, x, mode='valid')
        sum_yx_minus_x_squared[:-1, :-1] = z[1:, 1:]

        for i in range(n - d + 1):
            sum_yx_minus_x_squared[i, -1] = np.sum(x * y[i:i + d, n-d:])
            sum_yx_minus_x_squared[-1, i] = np.sum(x * y[n-d:, i:i + d])

        sum_yx_minus_x_squared = np.sum(np.square(x)) - 2 * sum_yx_minus_x_squared
        sum_yx_minus_x_squared *= - 0.5 / self._noise_std ** 2
        return sum_yx_minus_x_squared

    @nb.jit
    def _calc_prob_y_given_x_k_fast(self, y, x, k):
        n = y.shape[0]
        d = x.shape[0]

        max_k_in_row = min(n // d, k)

        ### Precomputing stuff
        log_size_S_per_row_per_k = np.zeros(max_k_in_row)
        for k_in_row in range(1, max_k_in_row + 1):
            log_size_S_per_row_per_k[k_in_row - 1] = LengthExtractor2D._compute_log_size_S_1d(n, k_in_row, d)

        # per viable pixel compute the prob the filter is there (Can be done faster using convolution)
        sum_yx_minus_x_squared = self._calc_sum_yx_minus_x_squared(y, x)

        # Per row compute the sum over all s_k
        pre_compute_per_row_per_k = np.zeros((n + 1, max_k_in_row))
        for j in range(n + 1-d):
            # Go to 1d case
            curr_sum_yx_minus_x_squared = sum_yx_minus_x_squared[j]
            mapping = np.full((n + 1, max_k_in_row + 1), -np.inf)
            mapping[:, 0] = 0
            for curr_k in range(1, max_k_in_row + 1):
                for i in range(n-d, -1, -1):
                    mapping[i, curr_k] = np.logaddexp(curr_sum_yx_minus_x_squared[i] + mapping[i + d, curr_k - 1],
                                                      mapping[i + 1, curr_k])
            pre_compute_per_row_per_k[j] = mapping[0, 1:]

        # Allocating memory
        # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
        # when k=0 the probability is 1
        mapping = np.full((n + 1, k + 1), -np.inf)
        mapping[:, 0] = 0

        # Filling values one by one, skipping irrelevant values
        mapping = np.full((n + 1, k + 1), -np.inf)
        mapping[:, 0] = 0
        for curr_k in range(1, k + 1):
            max_k = min(curr_k, max_k_in_row)
            for i in range(d, n+1):
                mapping[i, curr_k] = logsumexp(mapping[i - d, range(curr_k-1, curr_k-max_k-1, -1)] + pre_compute_per_row_per_k[i, :max_k])
                mapping[i, curr_k] = np.logaddexp(mapping[i, curr_k], mapping[i - 1, curr_k])
        very_well_separated_axis_1 = mapping[n, k]

        # Doing the same for y transpose to find very well separated on the other axis
        sum_yx_minus_x_squared = sum_yx_minus_x_squared.T.copy()
        pre_compute_per_row_per_k = np.zeros((n + 1, max_k_in_row))
        for j in range(n + 1-d):
            # Go to 1d case
            curr_sum_yx_minus_x_squared = sum_yx_minus_x_squared[j]
            mapping = np.full((n + 1, max_k_in_row + 1), -np.inf)
            mapping[:, 0] = 0
            for curr_k in range(1, max_k_in_row + 1):
                for i in range(n-d, -1, -1):
                    mapping[i, curr_k] = np.logaddexp(curr_sum_yx_minus_x_squared[i] + mapping[i + d, curr_k - 1],
                                                      mapping[i + 1, curr_k])
            pre_compute_per_row_per_k[j] = mapping[0, 1:]

        mapping = np.full((n + 1, k + 1), -np.inf)
        mapping[:, 0] = 0
        for curr_k in range(1, k + 1):
            max_k = min(curr_k, max_k_in_row)
            for i in range(d, n + 1):
                mapping[i, curr_k] = logsumexp(
                    mapping[i - d, range(curr_k - 1, curr_k - max_k - 1, -1)] + pre_compute_per_row_per_k[i, :max_k])
                mapping[i, curr_k] = np.logaddexp(mapping[i, curr_k], mapping[i - 1, curr_k])
        very_well_separated_axis_2 = mapping[n, k]

        very_well_separated_both_axis = np.logaddexp(very_well_separated_axis_1, very_well_separated_axis_2)

        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, k, d) - np.log(2)
        log_prob_all_noise = self.log_prob_all_noise

        likelihood = log_pd + log_prob_all_noise + very_well_separated_both_axis
        return likelihood

    def _calc_d_likelihood(self, y, d):
        tic = time.time()
        expected_k = self._find_expected_occurrences(y, d)

        signal_filter = self._signal_filter_gen(d)
        likelihood = self._calc_prob_y_given_x_k_fast(y, signal_filter, expected_k)
        toc = time.time()

        if self._logs:
            print(
                f"For D={d}, likelihood={likelihood}, Expected K={expected_k}, Time={toc - tic}")

        return likelihood

    def extract(self):
        likelihoods = [self._calc_d_likelihood(self._y, d) for d in self._length_options]
        d_best = self._length_options[np.argmax(likelihoods)]
        return likelihoods, d_best
