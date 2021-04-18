import numpy as np
from src.utils.utils import arange_data_2d

import matplotlib.pyplot as plt
from src.utils.utils import log_binomial
from scipy.special import logsumexp

np.random.seed(501)


class DynamicProgramming2D:
    columns = 100
    rows = 3
    occurrences = 2
    d_width = 10
    d_height = 10

    def __init__(self):
        pass

    def dp_single_row(self, occurrences):
        if occurrences == 0:
            return 1

        if occurrences * self.d_width > self.columns:
            return 0

        mapping = np.zeros(shape=(self.columns + 1, occurrences + 1))
        mapping[:, 0] = 1
        for k in np.arange(occurrences) + 1:
            for i in np.flip(np.arange(self.columns - self.d_width + 1)):
                num_of_available_cells = self.columns - i
                num_of_required_cells = k * self.d_width

                if num_of_required_cells > num_of_available_cells:
                    mapping[i, k] = 0
                else:
                    mapping[i, k] = mapping[i + 1, k] + mapping[i + self.d_width, k - 1]

        # a = np.exp(log_binomial(self.columns - (self.d_width - 1) * occurrences, occurrences))
        return mapping[0, occurrences]

    def dp_multilpe_rows(self):
        mapping = np.zeros(shape=(self.rows + 1, self.occurrences + 1))
        mapping[-1, 0] = 1
        for row in np.flip(np.arange(self.rows)):  # start from last row
            for k in np.arange(self.occurrences + 1):  # for each num of left occourences
                for _k in np.arange(k + 1):  # TODO: ~max{k, columns//d}
                    # _k go to current row, the rest goes to the rest
                    term1 = self.dp_single_row(_k)
                    term2 = mapping[row + 1, k - _k]
                    mapping[row, k] += term1 * term2
        print(mapping[0, self.occurrences])

    def log_dp_multilpe_rows(self):
        mapping = np.zeros(shape=(self.rows + 1, self.occurrences + 1))
        mapping[-1, 0] = 1
        for row in np.flip(np.arange(self.rows)):  # start from last row
            for k in np.arange(self.occurrences + 1):  # for each num of left occourences
                a = []
                for _k in np.arange(k + 1):  # TODO: ~max{k, columns//d}
                    # _k go to current row, the rest goes to the rest
                    term1 = log_binomial(self.columns - (self.d_width - 1) * _k, _k)
                    term2 = mapping[row + 1, k - _k]
                    a.append(term1 + term2)
                    # mapping[row, k] += term1 * term2
                mapping[row, k] = logsumexp(a)
        print(np.exp(mapping[0, self.occurrences]))
        print(mapping[0, self.occurrences])


# dp = DynamicProgramming2D()

# dp.dp_multilpe_rows()
# dp.log_dp_multilpe_rows()

rows = 1000
columns = 1000
d = 20
p = 1
k = 71
std = 0.1

data = arange_data_2d(rows=rows,
                      columns=columns,
                      signal_diameter=d,
                      signal_power=p,
                      occurrences=k,
                      noise_std=std,
                      noise_mean=0)

plt.imshow(data, cmap='gray')
plt.show()

# chuncked_rows = int(rows / d)
# chuncked_data = data.reshape(chuncked_rows, d, columns)
# downsample = np.sum(chuncked_data, 1).reshape(-1) / d

# plt.figure()
# plt.plot(downsample)
# plt.show()

# d_options = np.arange(1, d * 2, 5)
# le = LengthExtractor(y=downsample,
#                      length_options=d_options,
#                      signal_filter_gen=lambda d: np.full(d, 1),
#                      noise_mean=0,
#                      noise_std=std,
#                      signal_power_estimator_method=SignalPowerEstimator.SecondMoment,
#                      exp_attr=None,
#                      logs=True)
#
# likelihoods, d = le.extract()
# plt.figure()
# plt.plot(d_options, likelihoods)
# plt.show()

# plt.plot(data.sum(0))
# plt.show()
