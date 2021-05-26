import numpy as np
import numba as nb
import time
import logging
from enum import Enum
import src.algorithms.utils as utils

from src.algorithms.utils import log_binomial
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator as SPE


class SignalPowerEstimator(SPE):
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

        self.log_prob_all_noise = utils.log_prob_all_is_noise(self._data, self._noise_std)
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

    def _calc_length_likelihood(self, signal_filter, expected_num_of_occurrences):
        y = self._data
        n = self._data.shape[0]
        d = signal_filter.shape[0]

        sum_yx_minus_x_squared = utils.log_probability_filter_on_each_pixel(y, signal_filter, self._noise_std)
        mapping = utils.dynamic_programming_1d(n, expected_num_of_occurrences, d, sum_yx_minus_x_squared)

        # Computing remaining parts of log-likelihood
        log_pd = -utils.log_size_S_1d(n, expected_num_of_occurrences, d)
        log_prob_all_noise = self.log_prob_all_noise
        likelihood = log_pd + log_prob_all_noise + mapping[0, expected_num_of_occurrences]
        logging.debug(f'log pd: {log_pd}, noise: {log_prob_all_noise}, mapping:{mapping[0, expected_num_of_occurrences]}')
        return likelihood

    def _calc_signal_length_likelihood(self, d):
        tic = time.time()
        expected_num_of_signal_occurrences = self._estimate_num_of_signal_occurrences(d)

        signal_filter = self._signal_filter_gen(d)
        likelihood = self._calc_length_likelihood(signal_filter, expected_num_of_signal_occurrences)
        toc = time.time()

        if self._logs:
            logging.debug(
                f"For D={d}, likelihood={likelihood}, Expected K={expected_num_of_signal_occurrences}, Time={toc - tic}\n")

        return likelihood

    def estimate(self):
        likelihoods = [self._calc_signal_length_likelihood(d) for d in self._length_options]
        max_likelihood_length = self._length_options[np.argmax(likelihoods)]
        return likelihoods, max_likelihood_length
