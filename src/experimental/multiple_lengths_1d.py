import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.utils import create_random_k_tuple_sum_to_n
from src.algorithms.length_extractor_1d import LengthExtractor1D, SignalPowerEstimator
from src.experimental.length_extractor_1d_multiple_length import LengthExtractorML1D, SignalsDistribution
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore')


# np.random.seed(501)


def add_pulses(y, signal_mask, signal_gen):
    """
    Add to y the signal at positions signal_masl == 1
    """
    new_y = np.copy(y)
    s_cum = np.cumsum(signal_mask)
    for i in np.arange(s_cum.shape[0] - 1):
        signal = signal_gen()
        x_len = signal.shape[0]
        start = s_cum[i] + x_len * i
        new_y[start:start + x_len] = signal
    return new_y


def add_gaus_noise(y, mean, std):
    noise = np.random.normal(mean, std, y.shape[0])
    return y + noise


def signal_gen(ds, ds_dist, power):
    def signal():
        d = np.random.choice(ds, 1, ds_dist)
        return np.full(d, power)

    return signal


def simulate_data(n, ds, ds_dist, p, k, noise_std):
    d = np.max(ds)
    # signal = signal_gen(ds, ds_dist, p)
    y = np.zeros(n)
    signal_mask = create_random_k_tuple_sum_to_n(n - d * k, k + 1)
    pulses = add_pulses(y, signal_mask, signal_gen(ds, ds_dist, p))
    y = add_gaus_noise(pulses, 0, noise_std)
    return y, pulses


T = 1
noise_std = 1
n = 10000
k = 40
d = 100
cuts = np.array([0.4, 0.7, 1])
# ds_dist = [0.125, 0.325, 0.55]
ds_dist = [0.1, 0.8, 0.1]
ds = np.array(d * cuts, dtype=int)
p = 1,
signal_filter_gen = lambda d: np.full(d, 1)
y, pulses = simulate_data(n, ds, ds_dist, p, k, noise_std)

# plt.plot(y)
# plt.show()
length_options = np.arange(d // 4, int(d * 2), 5)
# length_options = [d]


signals_distributions = [SignalsDistribution(length=l, cuts=list(cuts), distribution=ds_dist,
                                             filter_gen=signal_filter_gen) for l in length_options]

likelihoods_1d = np.zeros_like(length_options)
likelihoods_ml1d = np.zeros_like(length_options)
for t in range(T):
    y, pulses = simulate_data(n, ds, ds_dist, p, k, noise_std)

    le = LengthExtractor1D(y=y,
                           length_options=length_options,
                           noise_std=noise_std)
    likelihoods, d = le.extract()
    likelihoods_1d = likelihoods_1d + np.array(likelihoods)

    le2 = LengthExtractorML1D(data=y,
                              length_distribution_options=signals_distributions, noise_std=noise_std)
    likelihoods2, d2 = le2.extract()
    likelihoods_ml1d = likelihoods_ml1d + np.array(likelihoods2)

likelihoods_ml1d /= T
likelihoods_1d /= T

plt.title(f'ML1d: {length_options[np.argmax(likelihoods_ml1d)]}, 1d: {length_options[np.argmax(likelihoods_1d)]}')
plt.plot(length_options, likelihoods_ml1d)
plt.plot(length_options, likelihoods_1d)
plt.show()
