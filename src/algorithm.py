import numpy as np
from src.utils import log_binomial
import numba as nb


class LengthExtractor:

    def __init__(self, y, length_options, signal_filter_gen, signal_seperation,
                 noise_mean, noise_std, exp_attr, logs=True):
        self._y = y
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._logs = logs
        self._exp_attr = exp_attr
        self._signal_seperation = signal_seperation

        self._n = self._y.shape[0]
        self._exact_signal_power = self._calc_signal_power(exp_attr["d"]) * exp_attr["k"]
        self.log_prob_all_noise = self._calc_log_prob_all_is_noise()

    def _calc_log_prob_all_is_noise(self):
        y = self._y
        n = y.shape[0]
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        return - n * np.log(self._noise_std * (2 * np.pi) ** 0.5) + minus_1_over_twice_variance * np.sum(np.square(y))

    def _calc_signal_power(self, d):
        return np.sum(np.power(self._signal_filter_gen(d), 2))

    def _find_expected_occurrences(self, y, d):

        if self._exp_attr["use_exact_k"]:
            return self._exp_attr["k"]

        # If we know the exact signal power we use it, else compute from data
        if self._exp_attr["use_exact_signal_power"]:
            all_signal_power = self._exact_signal_power
        else:
            y_power = np.sum(np.power(y, 2))
            noise_power = (self._noise_std ** 2 - self._noise_mean ** 2) * y.shape[0]
            all_signal_power = y_power - noise_power

        single_signal_power = self._calc_signal_power(d)
        k = int(np.round(all_signal_power / single_signal_power))
        return k

    def _compute_log_pd(self, n, k, d):
        """
        Compute log(1/|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping.
        """
        n_tag = n - (d - 1) * k
        k_tag = k
        return -log_binomial(n_tag, k_tag)

    def _calc_prob_y_given_x_k_slow(self, y, x, k):
        n = y.shape[0]
        d = x.shape[0]

        # Precomputing stuff
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        sum_yx_minus_x_squared = np.zeros(n - d + 1)
        x_squared = np.square(x)
        for i in range(n - d + 1):
            sum_yx_minus_x_squared[i] = np.sum(x_squared - 2 * x * y[i:i + d])

        sum_yx_minus_x_squared *= minus_1_over_twice_variance

        # Allocating memory
        mapping = np.zeros(shape=(n + 1, k + 1))

        def log_R(start_idx, num_signals):
            total_len = n - start_idx

            # If we don't need any more signals
            if num_signals == 0:
                return 0

            # If there is no legal way to put signals in the remaining space
            if total_len < num_signals * d:
                return -np.inf

            c1 = sum_yx_minus_x_squared[start_idx]
            return np.logaddexp(c1 + mapping[start_idx + d, num_signals - 1], mapping[start_idx + 1, num_signals])

        # Filling values one by one, skipping irrelevant values
        for i in np.arange(mapping.shape[0])[::-1]:
            for curr_k in np.arange(mapping.shape[1]):
                if i < (k - curr_k) * d:
                    continue
                mapping[i, curr_k] = log_R(i, curr_k)

        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, k, d)

        likelihood = log_pd + self.log_prob_all_noise + mapping[0, k]
        return likelihood

    @nb.jit
    def _calc_prob_y_given_x_k_fast(self, y, x, k):
        n = y.shape[0]
        d = x.shape[0]

        # Precomputing stuff
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        sum_yx_minus_x_squared = np.zeros(n - d + 1)
        x_squared = np.square(x)
        for i in range(n - d + 1):
            sum_yx_minus_x_squared[i] = np.sum(x_squared - 2 * x * y[i:i + d])

        sum_yx_minus_x_squared *= minus_1_over_twice_variance

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

        likelihood = log_pd + self.log_prob_all_noise + mapping[0, k]
        return likelihood

    def _calc_d_likelihood(self, y, d):
        expected_k = self._find_expected_occurrences(y, d)

        signal_with_sep_pad = np.pad(self._signal_filter_gen(d), [(0, self._signal_seperation)])
        likelihood = self._calc_prob_y_given_x_k_fast(y, signal_with_sep_pad, expected_k)

        if self._logs:
            print(f"For D={d - self._signal_seperation}, likelihood={likelihood}, Expected K={expected_k}")

        return likelihood

    def extract(self):
        likelihoods = [self._calc_d_likelihood(self._y, d) for d in self._length_options]
        d_best = self._length_options[np.argmax(likelihoods)]
        return likelihoods, d_best
