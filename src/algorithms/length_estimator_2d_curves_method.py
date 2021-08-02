import numpy as np
from skimage.draw import line

from src.algorithms.length_estimator_1d import LengthEstimator1D
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator
from src.utils.logger import logger
from src.algorithms.utils import calc_most_likelihood_and_optimized_power_1d


class LengthEstimator2DCurvesMethod:

    def __init__(self,
                 data,
                 length_options,
                 signal_filter_gen_1d,
                 noise_std,
                 noise_mean=0,
                 curve_width=31,
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen_1d
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._curve_width = curve_width
        self._logs = logs

        self._n = self._data.shape[0]
        self._num_of_curves = 30
        self._cut_fix_factor = 0.7
        self._fixed_num_of_occurrences = 3
        self._curves_noise = self._noise_std / np.sqrt(self._num_of_curves)

        self._curves = self._create_curves(num=self._num_of_curves)

        logger.info(f'Data length: {self._curves.shape[0]} x {self._curves.shape[1]}')

    def _rows_curves(self, jump=None):
        mat = self._data
        curve_rows = mat[np.arange(0, mat.shape[0], jump)]
        curve = np.concatenate(curve_rows)
        return curve

    def _random_lines_curves(self, num_of_curves):
        width = self._curve_width
        mat = self._data
        rows, columns = mat.shape
        width_buffer = (width - 1) // 2

        curves = []
        for _ in range(num_of_curves):
            side = np.random.randint(2)
            width_curve = []

            if side == 0:  # up to down line
                up_column = np.random.randint(width_buffer, columns - width_buffer)
                down_column = np.random.randint(width_buffer, columns - width_buffer)
                for i in range(width):
                    rr, cc = line(0, up_column - width_buffer + i, rows - 1, down_column - width_buffer + i)
                    width_curve.append(mat[rr, cc])

            else:  # left to right
                left_row = np.random.randint(width_buffer, rows - width_buffer)
                right_row = np.random.randint(width_buffer, rows - width_buffer)
                for i in range(width):
                    rr, cc = line(left_row - width_buffer + i, 0, right_row - width_buffer + i, columns - 1)
                    width_curve.append(mat[rr, cc])

            curve = np.nanmean(width_curve, axis=0)
            curves.append(curve)

        return curves

    def _create_curves(self, num, concat=1, high_power_selection_factor=2, strategy='random lines'):

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

    def estimate(self):
        fixed_length_options = (np.int32(self._length_options * self._cut_fix_factor))

        likelihoods = np.zeros(len(fixed_length_options))
        powers = np.zeros(len(fixed_length_options))
        for i, length in enumerate(fixed_length_options):
            likelihoods[i], powers[i] = calc_most_likelihood_and_optimized_power_1d(self._curves,
                                                                                    self._signal_filter_gen(length, 1),
                                                                                    self._fixed_num_of_occurrences,
                                                                                    self._curves_noise)
            logger.info(
                f'For length {self._length_options[i]} matched power is {powers[i]}, Likelihood={likelihoods[i]}')

        most_likely_length = self._length_options[np.nanargmax(likelihoods)]
        most_likely_power = powers[np.nanargmax(likelihoods)]
        return likelihoods, most_likely_length, most_likely_power
