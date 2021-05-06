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

np.random.seed(500)


def add_pulses(y, signal_mask, pulses):
    """
    Add to y the signal at positions signal_masl == 1
    """
    new_y = np.copy(y)
    start = 0
    for i in np.arange(signal_mask.shape[0] - 1):
        signal = pulses[i]
        x_len = signal.shape[0]
        start += signal_mask[i]
        new_y[start:start + x_len] = signal
        start += x_len
    return new_y


def add_gaus_noise(y, mean, std):
    noise = np.random.normal(mean, std, y.shape[0])
    return y + noise


def deterministic_signal_gen(d, power):
    return np.full(d, power)


def signal_gen(ds, ds_dist, power):
    def signal():
        d = np.random.choice(a=ds, size=1, p=ds_dist)
        return np.full(d, power)

    return signal


def simulate_data(n, ds, ds_dist, p, k, noise_std):
    y = np.zeros(n)

    occurrences_dist = np.array(k * np.array(ds_dist), dtype=int)
    total_length = np.sum(ds * occurrences_dist)

    pulses_types = np.concatenate([np.full(occurrences_dist[i], fill_value=i) for i in range(len(ds))])
    np.random.shuffle(pulses_types)
    pulses = [deterministic_signal_gen(ds[i], p) for i in pulses_types]
    signal_mask = create_random_k_tuple_sum_to_n(n - total_length, k + 1)
    pulses = add_pulses(y, signal_mask, pulses)

    y = add_gaus_noise(pulses, 0, noise_std)
    return y, pulses


noise_std = 3
n = 10000
k = 60
d = 50
cuts = np.array([0.25, 0.5, 1])
# ds_dist = [0.125, 0.325, 0.55]
ds_dist = [.0, .7, .3]
ds = np.array(d * cuts, dtype=int)
p = 1,
signal_filter_gen = lambda d: np.full(d, 1)
y, pulses = simulate_data(n, ds, ds_dist, p, k, noise_std)

# plt.plot(y)
# plt.show()
length_options = np.arange(d // 4, int(d * 2), 4)
# length_options = [64]


likelihoods_1d = np.zeros_like(length_options)
likelihoods_ml1d = np.zeros_like(length_options)

le = LengthExtractor1D(y=y,
                       length_options=length_options,
                       noise_std=noise_std)
likelihoods, d = le.extract()
likelihoods_1d = likelihoods_1d + np.array(likelihoods)

print('\n')

signals_distributions = [SignalsDistribution(length=l, cuts=list(cuts), distribution=ds_dist,
                                             filter_gen=signal_filter_gen) for l in length_options]
le2 = LengthExtractorML1D(data=y,
                          length_distribution_options=signals_distributions, noise_std=noise_std)
likelihoods2, d2 = le2.extract()
likelihoods_ml1d = likelihoods_ml1d + np.array(likelihoods2)

likelihoods_ml1d2 = np.zeros_like(length_options)
ds_dist2 = [.0, .3, .7]
signals_distributions2 = [SignalsDistribution(length=l, cuts=list(cuts), distribution=ds_dist2,
                                              filter_gen=signal_filter_gen) for l in length_options]
le3 = LengthExtractorML1D(data=y,
                          length_distribution_options=signals_distributions2, noise_std=noise_std)
likelihoods3, d3 = le3.extract()
likelihoods_ml1d2 = likelihoods_ml1d2 + np.array(likelihoods3)

likelihoods_ml1d4 = np.zeros_like(length_options)
ds_dist4 = [.0, .5, .5]
signals_distributions4 = [SignalsDistribution(length=l, cuts=list(cuts), distribution=ds_dist4,
                                              filter_gen=signal_filter_gen) for l in length_options]
le4 = LengthExtractorML1D(data=y,
                          length_distribution_options=signals_distributions4, noise_std=noise_std)
likelihoods4, d4 = le4.extract()
likelihoods_ml1d4 = likelihoods_ml1d4 + np.array(likelihoods4)

plt.title(f'ML1d: {length_options[np.argmax(likelihoods_ml1d)]}, 1d: {length_options[np.argmax(likelihoods_1d)]}\n'
          f'cuts: {cuts}, n={n}, d={d}, k={k}, std={noise_std}')
plt.plot(length_options, likelihoods_ml1d, label=f'multi {ds_dist}')
plt.plot(length_options, likelihoods_1d, label='single')
plt.plot(length_options, likelihoods_ml1d2, label=f'multi {ds_dist2}')
plt.plot(length_options, likelihoods_ml1d4, label=f'multi {ds_dist4}')
plt.legend(loc="upper right")
plt.show()
