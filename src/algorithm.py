import numpy as np


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
        nominator = np.sum(np.log(np.arange(k_tag) + 1)) + np.sum(np.log(np.arange(n_tag - k_tag) + 1))
        denominator = np.sum(np.log(np.arange(n_tag) + 1))
        return nominator - denominator

    def _calc_prob_y_given_x_k(self, y, x, k):
        n = y.shape[0]
        d = x.shape[0]

        # Precomputing stuff
        minus_1_over_twice_variance = - 0.5 / self._noise_std ** 2
        y_squared = np.square(y)
        sum_y_minus_x_squared = np.zeros(n - d + 1)
        for i in range(n - d + 1):
            sum_y_minus_x_squared[i] = np.sum(np.square(y[i:i + d] - x))

        case_1_const = minus_1_over_twice_variance * sum_y_minus_x_squared
        case_2_const = minus_1_over_twice_variance * y_squared

        case_2_const_cum_sum = np.zeros(n + 1)
        for i in range(1, n + 1):
            case_2_const_cum_sum[-i - 1] = case_2_const_cum_sum[-i] + case_2_const[-i]

        # Allocating memory
        mapping = np.zeros(shape=(n + 1, k + 1))

        def log_R(start_idx, num_signals):
            total_len = len(y) - start_idx

            # If we don't need any more signals
            if num_signals == 0:
                return case_2_const_cum_sum[start_idx]

            # If there is no legal way to put signals in the remaining space
            if total_len < num_signals * d:
                return -np.inf

            c1 = case_1_const[start_idx]
            c2 = case_2_const[start_idx]

            return np.logaddexp(c1 + mapping[start_idx + d, num_signals - 1], c2 + mapping[start_idx + 1, num_signals])

        # Filling values one by one, skipping irrelevant values
        for i in np.arange(mapping.shape[0])[::-1]:
            for curr_k in np.arange(mapping.shape[1]):
                if i < (k - curr_k) * d:
                    continue
                mapping[i, curr_k] = log_R(i, curr_k)

        # Computing remaining parts of log-likelihood
        log_pd = self._compute_log_pd(n, k, d)
        log_sigma_sqrt_2_pi = -np.log(self._noise_std * (2 * np.pi) ** 0.5)

        likelihood = log_pd + n * log_sigma_sqrt_2_pi + mapping[0, k]
        return likelihood

    def _calc_d_likelihood(self, y, d):
        expected_k = self._find_expected_occurrences(y, d)

        signal_with_sep_pad = np.pad(self._signal_filter_gen(d), [(0, self._signal_seperation)])
        likelihood = self._calc_prob_y_given_x_k(y, signal_with_sep_pad, expected_k)

        if self._logs:
            print(f"For D={d - self._signal_seperation}, likelihood={likelihood}, Expected K={expected_k}")

        return likelihood

    def extract(self):
        likelihoods = [self._calc_d_likelihood(self._y, d) for d in self._length_options]
        d_best = self._length_options[np.argmax(likelihoods)]
        return likelihoods, d_best
