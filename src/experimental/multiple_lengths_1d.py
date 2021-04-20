import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.utils import create_random_k_tuple_sum_to_n
from src.algorithms.length_extractor_1d import LengthExtractor1D, SignalPowerEstimator
from src.experimental.length_extractor_1d_multiple_length import LengthExtractorML1D, SignalsDistribution

np.random.seed(501)


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


noise_std = 1
n = 3000
k = 30
d = 30
cuts = np.array([1, 0.5, 0.3])
ds = np.array(d * cuts, dtype=int)
# ds_dist = [0.6, 0.3, 0.1]
ds_dist = [0.4, 0.3, 0.3]
p = 1,
signal_filter_gen = lambda d: np.full(d, 1)
y, pulses = simulate_data(n, ds, ds_dist, p, k, noise_std)

# plt.plot(y)
# plt.show()
length_options = np.arange(d // 4, int(d * 2), 5)
# length_options = [d]

le = LengthExtractor1D(y=y,
                       length_options=length_options,
                       noise_std=noise_std)
likelihoods, d = le.extract()

signals_distributions = [SignalsDistribution(length=l, cuts=list(cuts), distribution=ds_dist,
                                             filter_gen=signal_filter_gen) for l in length_options]
le2 = LengthExtractorML1D(data=y,
                          length_distribution_options=signals_distributions, noise_std=noise_std)
likelihoods2, d2 = le2.extract()

plt.plot(length_options, likelihoods)
plt.plot(length_options, likelihoods2)
plt.show()
