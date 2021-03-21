import numpy as np
from scipy.special import logsumexp

from utils import create_all_signal_mask_combs, add_pulses, Memoize


class LengthExtractor:

    def __init__(self, y, length_options, signal_avg_power, noise_mean, noise_std, logs=True):
        self._y = y
        self._length_options = length_options
        self._signal_avg_power = signal_avg_power
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._logs = logs

        self._n = self._y.shape[0]

    def _calc_length_likelihood(self, d):
        m = 1
        segments = np.array_split(self._y, m)
        d_likelihood = np.sum([self._calc_length_likelihood_segmented(segment, d) for segment in segments])
        return d_likelihood

    def _calc_length_likelihood_segmented(self, segment, d):
        n = segment.shape[0]
        k = self._find_expected_occourences(segment, d)
        signal = np.full(d, self._signal_avg_power)

        if self._logs:
            print(f'Calcing likelihood for segments of length {n}, d={d}')

        all_masks = create_all_signal_mask_combs(n, k, d)
        log_pd = - np.log(all_masks.shape[0])

        likelihoods = []
        for mask in all_masks:
            signal_hat = add_pulses(np.zeros(n), mask, signal)
            likelihoods.append(- np.sum((segment - signal_hat) ** 2 - self._noise_std ** 2))

        likelihood = log_pd + logsumexp(likelihoods)
        return likelihood

    def _find_expected_occourences(self, y, d):
        y_power = np.sum(np.power(y, 2))
        noise_power = (self._noise_std ** 2 - self._noise_mean ** 2) * y.shape[0]
        signal_power = y_power - noise_power
        return int(np.round(signal_power / d))

    def _compute_log_pd(self, n, k, d):
        n_tag = n - (d - 1) * k
        k_tag = k
        nominator = np.sum(np.log(np.arange(k_tag) + 1)) + np.sum(np.log(np.arange(n_tag - k_tag) + 1))
        denominator = np.sum(np.log(np.arange(n_tag) + 1))
        return nominator - denominator

    def _calc_d_likelihood_fast(self, segment, d):
        n = segment.shape[0]
        expected_k = self._find_expected_occourences(segment, d)
        signal = np.full(d, self._signal_avg_power)

        log_pd = self._compute_log_pd(n, expected_k, d)

        log_sigma_sqrt_2_pi = -np.log(self._noise_std * (2 * np.pi) ** 0.5)

        @Memoize
        def log_R(start, k):
            total_len = len(segment) - start

            # If we don't need any more signals
            if k == 0:
                return total_len * log_sigma_sqrt_2_pi - 0.5 * np.sum((segment[start:] / self._noise_std) ** 2)
            # If there is no legal way to put signals in the remaining space
            if total_len < k * d:
                return -np.inf
            # If there is only one legal way to put signals in the remaining space
            if total_len == d * k:
                return total_len * log_sigma_sqrt_2_pi - 0.5 * np.sum((segment[start:] - 1 / self._noise_std) ** 2)

            case_1_const = d * log_sigma_sqrt_2_pi - 0.5 * np.sum((segment[start:start + d] - 1 / self._noise_std) ** 2)
            case_2_const = log_sigma_sqrt_2_pi - 0.5 * (segment[start] / self._noise_std) ** 2

            return np.logaddexp(case_1_const + log_R(start + d, k - 1), case_2_const + log_R(start + 1, k))

        likelihood = log_pd + log_R(0, expected_k)
        return likelihood

    def extract(self):
        likelihoods = [self._calc_length_likelihood(d) for d in self._length_options]
        d_best = self._length_options[np.argmax(likelihoods)]
        return likelihoods, d_best

    def fast_extract(self):
        likelihoods = [self._calc_d_likelihood_fast(self._y, d) for d in self._length_options]
        d_best = self._length_options[np.argmax(likelihoods)]
        return likelihoods, d_best
