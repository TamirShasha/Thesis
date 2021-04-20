import numpy as np
import numba as nb
import time
from enum import Enum

from src.algorithms.utils import log_binomial
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator as SPE


class SignalPowerEstimator(SPE, Enum):
    Exact = "Exact Power"


class LengthExtractor1D:

    def __init__(self, y, length_options, noise_std,
                 noise_mean=0, signal_filter_gen=lambda l: np.full(l, 1),
                 signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
                 exp_attr=None, logs=True):
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
    def _compute_log_pd(n, k, d):
        """
        Compute log(1/|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping.
        """
        n_tag = n - (d - 1) * k
        k_tag = k
        return -log_binomial(n_tag, k_tag)

    @nb.jit
    def _calc_prob_y_given_x_k_fast(self, y, x, k):
        n = y.shape[0]
        d = x.shape[0]

        # Precomputing stuff
        sum_yx_minus_x_squared = np.zeros(n - d + 1)
        x_squared = np.square(x)
        for i in range(n - d + 1):
            sum_yx_minus_x_squared[i] = np.sum(x_squared - 2 * x * y[i:i + d])

        sum_yx_minus_x_squared *= - 0.5 / self._noise_std ** 2

        # Allocating memory
        # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
        # when k=0 the probability is 1
        mapping = np.full((n + 1, k + 1), -np.inf)
        mapping[:, 0] = 0

        # Filling values one by one, skipping irrelevant values
        # We already filled values when k=0 (=0) and when i>n-k*d
        # Values in  i<(k-curr_k)*d are not used for the computation of mapping[0,k]
        for curr_k in range(1, k + 1):
            for i in range(n - curr_k * d, (k - curr_k) * d - 1, -1):
                mapping[i, curr_k] = np.logaddexp(sum_yx_minus_x_squared[i] + mapping[i + d, curr_k - 1],
                                                  mapping[i + 1, curr_k])

        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, k, d)
        log_prob_all_noise = self.log_prob_all_noise
        likelihood = log_pd + log_prob_all_noise + mapping[0, k]
        print(mapping[0, k])
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
