import numpy as np
import numba as nb
import time
from enum import Enum

from src.algorithms.utils import log_binomial
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator as SPE


class SignalPowerEstimator(SPE, Enum):
    Exact = "Exact Power"


class LengthExtractor1D:

    def __init__(self, data, length_options, noise_std,
                 noise_mean=0,
                 signal_filter_gen=lambda l: np.full(l, 1),
                 signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
                 exp_attr=None,
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._exp_attr = exp_attr
        self._n = self._data.shape[0]

        self.log_prob_all_noise = self._calc_log_prob_all_is_noise()
        self._signal_power = self._estimate_full_signal_power()

    def _estimate_full_signal_power(self):
        # If we know the exact signal power we use it, else compute from data
        if self._signal_power_estimator_method == SignalPowerEstimator.Exact:
            signal_power = self._estimate_single_instance_of_signal_power(self._exp_attr["d"]) * self._exp_attr["k"]
        else:
            signal_power = estimate_signal_power(self._data, self._noise_std, self._noise_mean,
                                                 method=self._signal_power_estimator_method)
        return signal_power

    def _estimate_single_instance_of_signal_power(self, signal_length):
        return np.sum(self._signal_filter_gen(signal_length))

    def _estimate_num_of_signal_occurrences(self, signal_length):
        single_signal_power = self._estimate_single_instance_of_signal_power(signal_length)
        k = int(np.round(self._signal_power / single_signal_power))
        return k

    def _calc_log_prob_all_is_noise(self):
        if self._noise_std == 0:
            return 0

        y = self._data
        n = y.shape[0]
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        return - n * np.log(self._noise_std * (2 * np.pi) ** 0.5) + minus_1_over_twice_variance * np.sum(np.square(y))

    @staticmethod
    def _compute_log_pd(data_length, num_of_occurrences, signal_length):
        """
        Compute log(1/|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping.
        """
        n_tag = data_length - (signal_length - 1) * num_of_occurrences
        k_tag = num_of_occurrences
        return -log_binomial(n_tag, k_tag)

    # @nb.jit
    def _calc_length_likelihood(self, signal_filter, expected_num_of_occurrences):
        y = self._data
        n = self._data.shape[0]
        d = signal_filter.shape[0]

        # Precomputing stuff
        sum_yx_minus_x_squared = np.zeros(n - d + 1)
        x_squared = np.square(signal_filter)
        for i in range(n - d + 1):
            sum_yx_minus_x_squared[i] = np.sum(x_squared - 2 * signal_filter * y[i:i + d])
        sum_yx_minus_x_squared *= - 0.5 / self._noise_std ** 2

        # Allocating memory
        # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
        # when k=0 the probability is 1
        mapping = np.full((n + 1, expected_num_of_occurrences + 1), -np.inf)
        mapping[:, 0] = 0

        # Filling values one by one, skipping irrelevant values
        # We already filled values when k=0 (=0) and when i>n-k*d
        # Values in  i<(k-curr_k)*d are not used for the computation of mapping[0,k]
        for curr_k in range(1, expected_num_of_occurrences + 1):
            for i in range(n - curr_k * d, (expected_num_of_occurrences - curr_k) * d - 1, -1):
                mapping[i, curr_k] = np.logaddexp(sum_yx_minus_x_squared[i] + mapping[i + d, curr_k - 1],
                                                  mapping[i + 1, curr_k])

        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, expected_num_of_occurrences, d)
        log_prob_all_noise = self.log_prob_all_noise
        likelihood = log_pd + log_prob_all_noise + mapping[0, expected_num_of_occurrences]
        print(f'log pd: {log_pd}, noise: {log_prob_all_noise}, mapping:{mapping[0, expected_num_of_occurrences]}')
        return likelihood

    def _calc_signal_length_likelihood(self, d):
        tic = time.time()
        expected_num_of_signal_occurrences = self._estimate_num_of_signal_occurrences(d)

        signal_filter = self._signal_filter_gen(d)
        likelihood = self._calc_length_likelihood(signal_filter, expected_num_of_signal_occurrences)
        toc = time.time()

        if self._logs:
            print(
                f"For D={d}, likelihood={likelihood}, Expected K={expected_num_of_signal_occurrences}, Time={toc - tic}\n")

        return likelihood

    def estimate(self):
        likelihoods = [self._calc_signal_length_likelihood(d) for d in self._length_options]
        max_likelihood_length = self._length_options[np.argmax(likelihoods)]
        return likelihoods, max_likelihood_length
