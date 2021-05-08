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
        self.signals = np.array([filter_gen(l) for l in self.lengths], dtype=object)

    def _calc_avg_power(self):
        power = np.sum(self.filter_gen(self.length)) / self.length
        return power

    def _calc_avg_instance_power(self):
        return np.sum(self.lengths * self.power * self.distribution)

    def find_expected_occurrences(self, total_power):
        k = int(np.round(total_power / self.avg_instance_power))
        occurrences_dist = np.array(k * self.distribution, dtype=int)
        return occurrences_dist


class CircleCutsDistribution(SignalsDistribution):
    def __init__(self, length: int, filter_gen: Callable[[int], np.ndarray]):
        cuts = [0.4, 0.6, 1]
        distribution = [.0, .6, .4]
        super().__init__(length, cuts, distribution, filter_gen)


class Ellipse1t2CutsDistribution(SignalsDistribution):
    def __init__(self, length: int, filter_gen: Callable[[int], np.ndarray]):
        cuts = [.3, .5, 1]
        distribution = [.0, .8, 0.2]
        super().__init__(length, cuts, distribution, filter_gen)


class GeneralCutsDistribution(SignalsDistribution):
    def __init__(self, length: int, filter_gen: Callable[[int], np.ndarray]):
        cuts = [0.4, 0.7, 1]
        # distribution = [0.85, 0.1, 0.05]
        cuts = [1, 0.5, 0]
        distribution = [0.9, 0.1, 0]
        super().__init__(length, cuts, distribution, filter_gen)


class LengthExtractorML1D:

    def __init__(self,
                 data,
                 length_distribution_options: List[SignalsDistribution],
                 noise_std,
                 signal_separation=0,
                 noise_mean=0,
                 signal_power_estimator_method=SPE.FirstMoment,
                 logs=True):
        self._data = data
        self._length_distribution_options = length_distribution_options
        self._signal_separation = signal_separation
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._n = self._data.shape[0]

        self.log_prob_all_noise = self._estimate_log_prob_all_is_noise()

        self._signal_power = self._estimate_signal_power(self._data)

    def _estimate_signal_power(self, y):
        signal_power = estimate_signal_power(y, self._noise_std, self._noise_mean,
                                             method=self._signal_power_estimator_method)
        return signal_power

    def _estimate_log_prob_all_is_noise(self):
        y = self._data
        n = y.shape[0]
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        return - n * np.log(self._noise_std * (2 * np.pi) ** 0.5) + minus_1_over_twice_variance * np.sum(np.square(y))

    @staticmethod
    def _compute_log_pd(data_length, signals_lengths, signals_occurrences, signal_separation=0):
        """
        Compute log(1/|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping. And then coloring them with l colors such that k1,..,kl are the counts of each.
        """

        k = np.sum(signals_occurrences)  # total signals occurrences
        dk = np.sum((signals_lengths + signal_separation) * signals_occurrences)  # total signals length

        lb = np.sum(np.log(np.arange(1, data_length - dk + k + 1))) - \
             np.sum(np.log(np.arange(1, data_length - dk + 1)))

        for ki in signals_occurrences:
            lb -= np.sum(np.log(np.arange(1, ki + 1)))

        return -lb

    def _estimate_likelihood(self, signals_dist: SignalsDistribution):
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
        self._k = signals_dist.find_expected_occurrences(self._signal_power)  # k1, k2, .. , kl
        k = self._k

        mapping_length = np.max(lens) + self._signal_separation
        l1, l2, l3 = lens[0] + self._signal_separation, \
                     lens[1] + self._signal_separation, \
                     lens[2] + self._signal_separation
        shape = np.concatenate([[mapping_length + 1], np.array(k) + 2])

        mapping = np.full(shape, -np.inf)
        mapping[1, 0, 0, 0] = 0

        i_indices = [
            np.insert(((np.array([i, i + 1, i + l1, i + l2, i + l3]) - (n - 1)) % (mapping_length + 1)), 0, i)
            for i in np.arange(n - 1, -1, -1)]
        boundaries_list = [(0, k[0] + 1), (0, k[1] + 1), (0, k[2] + 1)]
        indices = np.array(list(itertools.product(*(range(*b) for b in boundaries_list))))

        k1, k2, k3 = indices[:, 0], indices[:, 1], indices[:, 2]

        for i, j, j_1, j_l1, j_l2, j_l3 in i_indices:
            tmp = [mapping[j_l1, k1 - 1, k2, k3],
                   mapping[j_l2, k1, k2 - 1, k3],
                   mapping[j_l3, k1, k2, k3 - 1],
                   mapping[j_1, k1, k2, k3]]
            tmp_map = np.stack(tmp, axis=-1).reshape(k[0] + 1, k[1] + 1, k[2] + 1, 4)
            tmp_map += sum_yx_minus_x_squared[np.newaxis, np.newaxis, [i], :]
            mapping[j, :-1, :-1, :-1] = logsumexp(tmp_map, axis=-1)

        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, signals_dist.lengths, self._k, self._signal_separation)
        log_prob_all_noise = self.log_prob_all_noise
        print(f'log pd: {log_pd}, noise: {log_prob_all_noise}, mapping:{mapping[j, k[0], k[1], k[2]]}')

        likelihood = log_pd + log_prob_all_noise + mapping[j, k[0], k[1], k[2]]
        return likelihood

    def _calc_length_distribution_likelihood(self, len_dist):
        tic = time.time()
        likelihood = self._estimate_likelihood(len_dist)
        toc = time.time()

        print(f'For d = {len_dist.lengths}, k = {self._k}, took {toc - tic} seconds, likelihood={likelihood}\n')

        return likelihood

    def extract(self):
        likelihoods = [self._calc_length_distribution_likelihood(ld) for ld in self._length_distribution_options]
        ld_best = self._length_distribution_options[np.argmax(likelihoods)]
        return likelihoods, ld_best
