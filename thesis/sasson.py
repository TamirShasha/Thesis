import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom, logsumexp

np.random.seed(500)

N = 50
D = 6
K = 2
NOISE_MEAN = 0
NOISE_STD = 1
SIGNAL_AVG_POWER = 1


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
def find_all_combinations(k, n):
    if k == 1:
        return np.array([n])

    combinations = []
    for i in np.arange(n + 1):
        tmp = find_all_combinations(k - 1, n - i)
        combinations.extend(np.column_stack((np.array([[i] * len(tmp)]).T, tmp)))
    return np.array(combinations)


def create_all_s_combs(signal_size, num_of_bars, size_of_bars):
    num_of_free_spots = signal_size - num_of_bars * size_of_bars
    print(f'creating all s combs for k = {num_of_bars + 1}, n = {num_of_free_spots}')
    return find_all_combinations(num_of_bars + 1, num_of_free_spots)


# Add to signal y k pulses of length d
def add_pulses(y, s, x):
    x_len = x.shape[0]
    s_cum = np.cumsum(s)
    for i in np.arange(s_cum.shape[0] - 1):
        start = s_cum[i] + x_len * i
        y[start:start + x_len] = x
    return y


def add_noise(y):
    noise = np.random.normal(NOISE_MEAN, NOISE_STD, y.shape[0])
    return y + noise


def find_k(y, d):
    y_power = np.sum(np.power(y, 2))
    noise_power = (NOISE_STD ** 2 - NOISE_MEAN ** 2) * y.shape[0]
    signal_power = y_power - noise_power
    return int(np.round(signal_power / d))


def calc_pd(n, k, d):
    return 1 / binom(n - (d - 1) * k, k)


def calc_d_likelihood(y, d):
    M = 1
    print(f'Seperated into {M} Segments')
    segments = np.array_split(y, M)
    d_likelihood = np.prod([calc_segment_likelihood(segment, d) for segment in segments])
    return d_likelihood


def calc_segment_likelihood(segment, d):
    n = segment.shape[0]
    k = find_k(segment, d)
    x = create_const_signal(SIGNAL_AVG_POWER, d)

    all_s = create_all_s_combs(n, k, d)
    log_pd = - np.log(all_s.shape[0])

    print(f'D: {d}, K: {k}, S Size: {all_s.shape[0]}')

    all_likelihoods = []
    for s in all_s:
        x_hat = add_pulses(np.zeros(n), s, x)
        all_likelihoods.append(- np.sum((segment - x_hat) ** 2 - NOISE_STD ** 2))

    likelihood = log_pd + logsumexp(all_likelihoods)
    print(likelihood)
    return likelihood


def find_d(y, d_options):
    liklihoods = [calc_d_likelihood(y, d) for d in d_options]
    print(liklihoods)
    d_best = d_options[np.argmax(liklihoods)]
    return liklihoods, d_best


def create_random_s(n, total_sum):
    if n == 1:
        return [total_sum]

    head = np.random.randint(0, (total_sum + 1) / n)
    tail = create_random_s(n - 1, total_sum - head)
    return np.concatenate(([head], tail))


def create_const_signal(avg_power, length):
    return np.full(length, avg_power)


y = np.zeros(N)

s = create_random_s(K + 1, N - K * D)
x = create_const_signal(SIGNAL_AVG_POWER, D)
y_clean = add_pulses(y, s, x)
y = add_noise(y_clean)

_d_options = np.arange(1, D * 5)
liklihoods, d = find_d(y, _d_options)
print(f'D Best: {d}')

plt.plot(y_clean)
plt.show()

plt.plot(_d_options, liklihoods)
plt.show()
