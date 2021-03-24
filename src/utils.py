import numpy as np


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]


@Memoize
def _find_all_signal_mask_combinations(k, n):
    if k == 1:
        return np.array([n])

    combinations = []
    for i in np.arange(n + 1):
        tmp = _find_all_signal_mask_combinations(k - 1, n - i)
        combinations.extend(np.column_stack((np.array([[i] * len(tmp)]).T, tmp)))
    return np.array(combinations)


def create_all_signal_mask_combs(signal_size, num_of_bars, size_of_bars):
    num_of_free_spots = signal_size - num_of_bars * size_of_bars
    return _find_all_signal_mask_combinations(num_of_bars + 1, num_of_free_spots)


def create_random_k_tuple_sum_to_n(n, k):
    """
    Output a random k tuple of non-negative integers that sums to n, with uniform probability over all options.
    """
    if k == 1:
        return [n]

    # Probably prob_x0_is_i = 1 / (1 + n - i) is true, need to check
    log_num_options_x0_is_i = np.zeros(n + 1)
    for i in range(n + 1):
        log_num_options_x0_is_i[i] = log_num_k_sums_to_n(n - i, k - 1)
    log_num_options_total = log_num_k_sums_to_n(n, k)
    prob_x0_is_i = np.exp(log_num_options_x0_is_i - log_num_options_total)
    head = np.random.choice(np.arange(n + 1), p=prob_x0_is_i)
    tail = create_random_k_tuple_sum_to_n(n - head, k - 1)
    return np.concatenate(([head], tail))


def log_num_k_sums_to_n(n, k):
    """
    Compute the log number of #{k tuples that sum to n}.
    """
    n_tag = n + k - 1
    k_tag = k - 1
    return log_binomial(n_tag, k_tag)


def log_binomial(n, k):
    """
    Compute the log of the binomial coefficient.
    """
    nominator = np.sum(np.log(np.arange(n) + 1))
    denominator = np.sum(np.log(np.arange(k) + 1)) + np.sum(np.log(np.arange(n - k) + 1))
    return nominator - denominator


# Add to signal y k pulses of length d
def add_pulses(y, signal_mask, signal):
    new_y = np.copy(y)
    x_len = signal.shape[0]
    s_cum = np.cumsum(signal_mask)
    for i in np.arange(s_cum.shape[0] - 1):
        start = s_cum[i] + x_len * i
        new_y[start:start + x_len] = signal
    return new_y


def add_gaus_noise(y, mean, std):
    noise = np.random.normal(mean, std, y.shape[0])
    return y + noise
