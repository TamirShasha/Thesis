import itertools
import numpy as np
from src.experimental.length_extractor_1d_multiple_length import LengthExtractorML1D, SignalsDistribution
from src.experiments.data_simulator_1d import simulate_data

# boundaries_list = [(0, 4, 1)] + [(5, 1, -1)]
# for values in itertools.product(*(range(*b) for b in boundaries_list)):
#     if np.all(values == 1):
#         print('all 0')
#     print(values)
# # do things with the values tuple, do_whatever(*values) perhaps


data, pulses = simulate_data(100, 10, 1, 3, 0.5)
ld = [SignalsDistribution(d, [1, 0.5], [0.7, 0.3], lambda d: np.full(d, 1)) for d in [5, 10, 15]]
r = LengthExtractorML1D(data, ld, 2).extract()
