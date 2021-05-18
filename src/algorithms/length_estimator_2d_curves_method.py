import numpy as np
from skimage.draw import line
from src.algorithms.length_estimator_1d import LengthExtractor1D
from src.experimental.length_extractor_1d_multiple_length import LengthExtractorML1D, CircleCutsDistribution, \
    Ellipse1t2CutsDistribution, SignalsDistribution
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator


class LengthExtractor2DCurvesMethod:

    def __init__(self,
                 data,
                 length_options,
                 signal_filter_gen,
                 noise_mean,
                 noise_std,
                 signal_power_estimator_method,
                 exp_attr,
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._exp_attr = exp_attr
        self._n = self._data.shape[0]

        self._num_of_curves = 2 * int(np.log(np.max(self._data.shape)))
        self._curves = self._create_curves(num=self._num_of_curves,
                                           high_power_selection_factor=100,
                                           concat=5)
        print(f'data length: {self._curves.shape[0]} x {self._curves.shape[1]}')

    def _rows_curves(self, jump=None):
        mat = self._data
        curve_rows = mat[np.arange(0, mat.shape[0], jump)]
        curve = np.concatenate(curve_rows)
        return curve

    def _random_lines_curves(self, num_of_curves):
        mat = self._data
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

        return curves

    def _create_curves(self, num, concat=2, high_power_selection_factor=100, strategy='random lines'):

        if strategy == 'rows':
            jump = self._data.shape[0] // num
            return self._rows_curves(jump)

        if strategy == 'random lines':
            curves = self._random_lines_curves(high_power_selection_factor * num)
            curves_powers = [
                estimate_signal_power(curve, self._noise_std, self._noise_mean, SignalPowerEstimator.FirstMoment)
                for curve in curves]
            top_curves = np.array(curves)[np.argsort(curves_powers)[-num * concat:]]
            top_concatenated_curves = np.array([np.concatenate(x) for x in np.split(top_curves, num)])
            return top_concatenated_curves

    def _estimate_likelihood_of_distribution(self, signals_distributions, sep=0):
        print(f'Will average over {self._num_of_curves} runs (curves)')

        # data = np.concatenate(self._create_1d_data_from_curves(num_of_curves))
        data = self._curves

        best_lengths = []
        sum_likelihoods = np.zeros_like(self._length_options)
        for t in range(self._num_of_curves):
            print(f'At iteration {t + 1}')
            likelihoods, d = LengthExtractorML1D(data=data[t],
                                                 length_distribution_options=signals_distributions,
                                                 noise_std=self._noise_std,
                                                 signal_separation=sep).extract()
            sum_likelihoods = sum_likelihoods + likelihoods
            curr_best_length = self._length_options[np.argmax(sum_likelihoods / (t + 1))]
            best_lengths.append(curr_best_length)

        likelihoods = sum_likelihoods / self._num_of_curves

        return likelihoods, best_lengths

    def _estimate_likelihood_for_1d(self):
        print(f'Will average over {self._num_of_curves} runs (curves)')

        best_lengths = []
        sum_likelihoods = np.zeros_like(self._length_options)
        for t in range(self._num_of_curves):
            print(f'At iteration {t}')
            likelihoods, best_len = LengthExtractor1D(data=self._curves[t],
                                                      length_options=self._length_options,
                                                      signal_filter_gen=self._signal_filter_gen,
                                                      noise_mean=self._noise_mean,
                                                      noise_std=self._noise_std,
                                                      signal_power_estimator_method=self._signal_power_estimator_method,
                                                      exp_attr=self._exp_attr,
                                                      logs=self._logs).estimate()
            sum_likelihoods = sum_likelihoods + likelihoods
            curr_best_length = self._length_options[np.argmax(sum_likelihoods / (t + 1))]
            best_lengths.append(curr_best_length)

        likelihoods = sum_likelihoods / self._num_of_curves

        return likelihoods, best_lengths

    def estimate(self, distributions=('circle', 'ellipse12', '1d')):
        likelihoods = dict()
        best_lengths = dict()

        if 'circle' in distributions:
            print(f'Running for circle distribution')
            signals_distributions = [CircleCutsDistribution(length=l, filter_gen=self._signal_filter_gen)
                                     for l in self._length_options]
            circle_likelihoods, circle_best_ds = self._estimate_likelihood_of_distribution(signals_distributions, 0)
            likelihoods['circle'] = circle_likelihoods
            best_lengths['circle'] = circle_best_ds

        if 'ellipse12' in distributions:
            print(f'Running for ellipse 1:2 distribution')
            signals_distributions = [Ellipse1t2CutsDistribution(length=l, filter_gen=self._signal_filter_gen)
                                     for l in self._length_options]
            ellipse_likelihoods, ellipse_best_lengths = self._estimate_likelihood_of_distribution(signals_distributions, 0)
            likelihoods['ellipse12'] = ellipse_likelihoods
            best_lengths['ellipse12'] = ellipse_best_lengths

        if '1d' in distributions:
            print('Running for 1d')
            one_d_likelihoods, one_d_best_lengths = self._estimate_likelihood_for_1d()
            likelihoods['one_dim'] = one_d_likelihoods
            best_lengths['one_dim'] = one_d_best_lengths

        return likelihoods, best_lengths
