import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
from src.algorithms.length_extractor_1d import LengthExtractor1D
from src.experimental.length_extractor_1d_multiple_length import LengthExtractorML1D, CircleCutsDistribution


class LengthExtractor2D:

    def __init__(self,
                 y,
                 length_options,
                 signal_filter_gen,
                 noise_mean,
                 noise_std,
                 signal_power_estimator_method,
                 exp_attr,
                 logs=True):
        self._y = y
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._exp_attr = exp_attr
        self._n = self._y.shape[0]

    def _rows_curves(self, jump=None):
        mat = self._y
        curve_rows = mat[np.arange(0, mat.shape[0], jump)]
        curve = np.concatenate(curve_rows)
        return curve

    def _random_lines_curves(self, num_of_curves):
        mat = self._y
        rows, columns = mat.shape

        curves = []
        for _ in range(num_of_curves):
            side = np.random.randint(2)
            if side == 0:  # up to down line
                up_column = np.random.randint(columns)
                down_column = np.random.randint(columns)
                rr, cc = line(0, up_column, rows - 1, down_column)
            else:  # left to right
                left_row = np.random.randint(rows)
                right_row = np.random.randint(rows)
                rr, cc = line(left_row, 0, right_row, columns - 1)

            curves.append(mat[rr, cc])

        curve = np.concatenate(curves)
        return curve

    def _create_1d_data_from_curves(self, num_of_curves, strategy='random lines'):
        if strategy == 'rows':
            jump = self._y.shape[0] // num_of_curves
            return self._rows_curves(jump)

        if strategy == 'random lines':
            return self._random_lines_curves(num_of_curves)

    def extract2(self):
        signals_distributions = [CircleCutsDistribution(length=l, filter_gen=self._signal_filter_gen)
                                 for l in self._length_options]
        num_of_curves = int(np.log(np.max(self._y.shape)))
        times = int(np.log(self._y.shape[0]))
        times = 3
        print(f'Will average over {times} runs, Num of curves is: {num_of_curves}')

        sum_likelihoods = np.zeros_like(self._length_options)
        for t in range(times):
            print(f'At iteration {t}')
            data = self._create_1d_data_from_curves(num_of_curves)
            likelihoods, d = LengthExtractorML1D(data=data,
                                                 length_distribution_options=signals_distributions,
                                                 noise_std=self._noise_std).extract()
            sum_likelihoods = sum_likelihoods + likelihoods

        likelihoods = sum_likelihoods / times

        return likelihoods

    def extract(self):
        # scan over the data to create 1d data
        y_1d = self._create_1d_data_from_curves(1)

        print(f'data 1d length is {y_1d.shape[0]}')

        # create length extractor
        length_extractor_1d = LengthExtractor1D(y=y_1d,
                                                length_options=self._length_options,
                                                signal_filter_gen=self._signal_filter_gen,
                                                noise_mean=self._noise_mean,
                                                noise_std=self._noise_std,
                                                signal_power_estimator_method=self._signal_power_estimator_method,
                                                exp_attr=self._exp_attr,
                                                logs=self._logs)
        likelihoods, d = length_extractor_1d.extract()
        return likelihoods, d
