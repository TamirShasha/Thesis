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


def create_random_signal_mask(n, total_sum):
    if n == 1:
        return [total_sum]

    head = np.random.randint(0, (total_sum + 1) / n + 1)
    tail = create_random_signal_mask(n - 1, total_sum - head)
    return np.concatenate(([head], tail))


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
