import numpy as np
import numba as nb
import time
from typing import List, Callable
import itertools
from src.utils.logsumexp import logsumexp
# from scipy.special import logsumexp

from src.algorithms.utils import log_binomial
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator as SPE


class SignalsDistribution:
    def __init__(self,
                 length: int,
                 cuts: list,
                 distribution: list,
                 filter_gen: Callable[[int], np.ndarray]):
        self.length = length
        self.cuts = np.array(cuts)
        self.distribution = np.array(distribution)
        self.filter_gen = filter_gen

        self.lengths = np.array(length * self.cuts, dtype=int)
        self.power = self._calc_avg_power()
        self.avg_instance_power = self._calc_avg_instance_power()
        self.signals = np.array([filter_gen(l) for l in self.lengths])

    def _calc_avg_power(self):
        power = np.sum(self.filter_gen(self.length)) / self.length
        return power

    def _calc_avg_instance_power(self):
        return np.sum(self.lengths * self.power * self.distribution)

    def find_expected_occurrences(self, total_power):
        k = int(np.round(total_power / self.avg_instance_power))
        occurrences_dist = np.array(k * self.distribution, dtype=int)
        return occurrences_dist


class LengthExtractorML1D:

    def __init__(self,
                 data,
                 length_distribution_options: List[SignalsDistribution],
                 noise_std,
                 noise_mean=0,
                 signal_power_estimator_method=SPE.FirstMoment,
                 logs=True):
        self._data = data
        self._length_distribution_options = length_distribution_options
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._n = self._data.shape[0]

        self.log_prob_all_noise = self._calc_log_prob_all_is_noise()

        self._signal_power = self._calc_signal_power(self._data)

    def _calc_signal_power(self, y):
        signal_power = estimate_signal_power(y, self._noise_std, self._noise_mean,
                                             method=self._signal_power_estimator_method)
        return signal_power

    def _calc_log_prob_all_is_noise(self):
        y = self._data
        n = y.shape[0]
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        return - n * np.log(self._noise_std * (2 * np.pi) ** 0.5) + minus_1_over_twice_variance * np.sum(np.square(y))

    @staticmethod
    def _compute_log_pd(data_length, signals_lengths, signals_occurrences):
        """
        Compute log(1/|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping. And then coloring them with l colors such that k1,..,kl are the counts of each.
        """

        k = np.sum(signals_occurrences)  # total signals occurrences
        dk = np.sum(signals_lengths * signals_occurrences)  # total signals length

        lb = np.sum(np.log(np.arange(1, data_length - dk + k + 1))) - \
             np.sum(np.log(np.arange(1, data_length - dk + 1)))

        for ki in signals_occurrences:
            lb -= np.sum(np.log(np.arange(1, ki + 1)))

        return -lb

    # @nb.jit
    def _calc_likelihood_fast(self, signals_dist: SignalsDistribution):
        y = self._data
        n = y.shape[0]
        lens = signals_dist.lengths
        nk = len(lens)

        # Precomputing stuff
        sum_yx_minus_x_squared = np.zeros([n, nk])
        signals_squared = np.square(signals_dist.signals)
        for i in range(n):
            for j, d in enumerate(lens):
                if i + d > n - 1:
                    continue
                sum_yx_minus_x_squared[i, j] = np.sum(signals_squared[j] - 2 * signals_dist.signals[j] * y[i: i + d])

        sum_yx_minus_x_squared *= - 0.5 / self._noise_std ** 2

        # Allocating memory
        # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
        # when k=0 the probability is 1

        k = signals_dist.find_expected_occurrences(self._signal_power)  # k1, k2, .. , kl
        shape = np.concatenate([[n + 1], np.array(k) + 1])
        mapping = np.full(shape, -np.inf)
        # mapping[:, 0] = 0

        boundaries_list = [(0, ki + 1, 1) for ki in k] + [(n - 1, -1, -1)]
        for indices in itertools.product(*(range(*b) for b in boundaries_list)):
            ks = list(indices[:-1])
            i = indices[-1]
            curr_loc = tuple([i] + ks)

            if np.sum(ks) == 0:
                mapping[curr_loc] = 0
                continue

            options_values = np.full(nk, -np.inf)
            for j in range(nk):  # if current i has k'th length starting at it
                if ks[j] == 0 or i + lens[j] > n - 1:
                    continue

                next_ks = list.copy(ks)
                next_ks[j] -= 1
                next_loc = tuple([i + lens[j]] + next_ks)
                options_values[j] = sum_yx_minus_x_squared[i, j] + mapping[next_loc]

            val = np.logaddexp(logsumexp(options_values), mapping[tuple([i + 1] + ks)])
            mapping[curr_loc] = val

        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, signals_dist.lengths, k)
        log_prob_all_noise = self.log_prob_all_noise

        start_loc = tuple([0] + list(k))
        likelihood = log_pd + log_prob_all_noise + mapping[start_loc]
        return likelihood

    def _calc_likelihood_fast3(self, signals_dist: SignalsDistribution):
        y = self._data
        n = y.shape[0]
        lens = signals_dist.lengths

        # Precomputing stuff
        sum_yx_minus_x_squared = np.zeros([n, 4])
        signals_squared = np.square(signals_dist.signals)
        for i in range(n):
            for j, d in enumerate(lens):
                if i + d > n - 1:
                    continue
                sum_yx_minus_x_squared[i, j] = np.sum(signals_squared[j] - 2 * signals_dist.signals[j] * y[i: i + d])

        sum_yx_minus_x_squared *= - 0.5 / self._noise_std ** 2
        k = signals_dist.find_expected_occurrences(self._signal_power)  # k1, k2, .. , kl
        R = self.tmp(n, signals_dist.length, k, sum_yx_minus_x_squared, lens)
        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, signals_dist.lengths, k)
        log_prob_all_noise = self.log_prob_all_noise

        start_loc = tuple([0] + list(k))
        # likelihood = log_pd + log_prob_all_noise + mapping[start_loc]
        likelihood = log_pd + log_prob_all_noise + R
        return likelihood

    @nb.jit
    def tmp(self, n, d, k, sum_yx_minus_x_squared, lens):
        shape = np.concatenate([[n + 1 + d], np.array(k) + 2])
        mapping = np.full(shape, -np.inf)
        mapping[:, 0, 0, 0] = 0

        tmp_map = np.zeros((k[0] + 1, k[1] + 1, k[2] + 1, 4))
        print(tmp_map.shape)
        for i in range(n - 1, -1, -1):
            for k1 in range(0, k[0] + 1, 1):
                for k2 in range(0, k[1] + 1, 1):
                    for k3 in range(0, k[2] + 1, 1):
                        tmp_map[k1, k2, k3] = [mapping[i + lens[0], k1 - 1, k2, k3],
                                               mapping[i + lens[1], k1, k2 - 1, k3],
                                               mapping[i + lens[2], k1, k2, k3 - 1],
                                               mapping[i + 1, k1, k2, k3]]
            tmp_map += sum_yx_minus_x_squared[np.newaxis, np.newaxis, [i], :]
            mapping[i, :-1, :-1, :-1] = logsumexp(tmp_map, axis=-1)
        return mapping[0, k[0], k[1], k[2]]

    def _calc_length_distribution_likelihood(self, len_dist):
        tic = time.time()
        likelihood = self._calc_likelihood_fast3(len_dist)
        toc = time.time()

        print(f'For d = {len_dist.length} took {toc - tic} seconds, likelihood={likelihood}')

        return likelihood

    def extract(self):
        likelihoods = [self._calc_length_distribution_likelihood(ld) for ld in self._length_distribution_options]
        ld_best = self._length_distribution_options[np.argmax(likelihoods)]
        return likelihoods, ld_best
