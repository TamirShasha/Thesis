import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.utils import create_random_k_tuple_sum_to_n
from src.algorithms.length_extractor_1d import LengthExtractor1D, SignalPowerEstimator


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
n = 70000
k = 200
ds = [150, 100]
ds_dist = [0.3, 0.7]
p = 1,
y, pulses = simulate_data(n, ds, ds_dist, p, k, noise_std)

le = LengthExtractor1D(y=y,
                       length_options=np.arange(10, int(ds[0] * 1.3), 5),
                       signal_filter_gen=lambda d: np.full(d, 1),
                       noise_mean=0,
                       noise_std=noise_std,
                       signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
                       exp_attr={},
                       logs=True)
likelihoods, d = le.extract()

plt.plot(np.arange(10, int(ds[0] * 1.3), 5), likelihoods)
plt.show()
