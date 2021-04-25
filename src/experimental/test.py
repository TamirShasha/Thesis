import itertools
import numpy as np
from src.experimental.length_extractor_1d_multiple_length import LengthExtractorML1D, SignalsDistribution
from src.algorithms.length_extractor_1d import LengthExtractor1D
from src.experiments.data_simulator_1d import simulate_data
import matplotlib.pyplot as plt

np.random.seed(501)

n, d, p, k, noise_std = 1000, 70, 1, 3, 1
signal_filter_gen = lambda l: np.full(l, 1)

data, pulses = simulate_data(n, d, p, k, noise_std)
length_options = np.arange(d // 2, int(d * 1.3), 2)
# length_options = [35]
ld = [SignalsDistribution(l, [1, 0], [1, 0], signal_filter_gen) for l in length_options]
likelihoods, best_ld = LengthExtractorML1D(data, ld, noise_std).extract()
print(f'best length dist is: {best_ld.length}')

le = LengthExtractor1D(y=data, length_options=length_options, signal_filter_gen=signal_filter_gen, noise_std=noise_std)
likelihoods2, d_best = le.extract()

plt.plot(length_options, likelihoods2)
plt.plot(length_options, likelihoods)
plt.show()

# v = LengthExtractorML1D._compute_log_pd(1000, np.array([35, 0]), np.array([6, 0]))
# print(v)
#
# from src.algorithms.utils import log_binomial
# v2 = log_binomial(1000 - 34*6, 6)
# print(v2)

