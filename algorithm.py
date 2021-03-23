import numpy as np


class LengthExtractor:

    def __init__(self, y, length_options, signal_avg_power, signal_seperation, noise_mean, noise_std, exp_attr,
                 logs=True):
        self._y = y
        self._length_options = length_options
        self._signal_avg_power = signal_avg_power
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._logs = logs
        self._exp_attr = exp_attr
        self._signal_seperation = signal_seperation

        self._n = self._y.shape[0]
        self._exact_signal_power = self._exp_attr["d"] * exp_attr["k"] * signal_avg_power

    def _find_expected_occourences(self, y, d):

        if self._exp_attr["use_exact_signal_power"]:
            return int(np.round(self._exact_signal_power / (d * self._signal_avg_power)))

        y_power = np.sum(np.power(y, 2))
        noise_power = (self._noise_std ** 2 - self._noise_mean ** 2) * y.shape[0]
        signal_power = y_power - noise_power
        k = int(np.round(signal_power / d))

        return k

    def _compute_log_pd(self, n, k, d):
        n_tag = n - (d - 1) * k
        k_tag = k
        nominator = np.sum(np.log(np.arange(k_tag) + 1)) + np.sum(np.log(np.arange(n_tag - k_tag) + 1))
        denominator = np.sum(np.log(np.arange(n_tag) + 1))
        return nominator - denominator

    def _calc_prob_dynamicly(self, mapping, segment, i, k, d, log_sigma_sqrt_2_pi):
        total_len = len(segment) - i

        # If we don't need any more signals
        if k == 0:
            return total_len * log_sigma_sqrt_2_pi - 0.5 * np.sum((segment[i:] / self._noise_std) ** 2)
        # If there is no legal way to put signals in the remaining space
        if total_len < k * d:
            return -np.inf
        # If there is only one legal way to put signals in the remaining space
        if total_len == d * k:
            return total_len * log_sigma_sqrt_2_pi - 0.5 * np.sum((segment[i:] - 1 / self._noise_std) ** 2)

        case_1_const = d * log_sigma_sqrt_2_pi - 0.5 * np.sum((segment[i:i + d] - 1 / self._noise_std) ** 2)
        case_2_const = log_sigma_sqrt_2_pi - 0.5 * (segment[i] / self._noise_std) ** 2

        return np.logaddexp(case_1_const + mapping[i + d, k - 1], case_2_const + mapping[i + 1, k])

    def _calc_d_likelihood(self, segment, d):
        n = segment.shape[0]
        expected_k = self._find_expected_occourences(segment, d)

        d_s = d + self._signal_seperation
        log_pd = self._compute_log_pd(n, expected_k, d_s)
        log_sigma_sqrt_2_pi = -np.log(self._noise_std * (2 * np.pi) ** 0.5)
        mapping = np.zeros(shape=(n, expected_k + 1))

        for i in np.arange(n)[::-1]:
            for k in np.arange(expected_k + 1):
                val = self._calc_prob_dynamicly(mapping, segment, i, k, d_s, log_sigma_sqrt_2_pi)
                mapping[i, k] = val

        likelihood = log_pd + mapping[0, expected_k]
        # likelihood = mapping[0, expected_k]

        if self._logs:
            print(f"For D={d}, likelihood={likelihood}, Expected K={expected_k}")

        return likelihood

    def extract(self):
        likelihoods = [self._calc_d_likelihood(self._y, d) for d in self._length_options]
        d_best = self._length_options[np.argmax(likelihoods)]
        return likelihoods, d_best
