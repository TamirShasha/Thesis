import numpy as np
from skimage.draw import line, bezier_curve, disk

from src.algorithms.length_estimator_1d import LengthEstimator1D
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator
from src.utils.logger import logger
from src.algorithms.utils import max_argmax_1d_case


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
        self._num_of_curves = 50
        self._cut_fix_factor = 0.7
        self._fixed_num_of_occurrences = 3
        # = self._noise_std / np.sqrt(self._num_of_curves)

        self._curves, self._curves_noise = self._create_curves(num=self._num_of_curves, split=1)

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

    def _split_image_into_blocks(self, grid_size=1):
        rows = self._data.shape[0] - self._data.shape[0] % grid_size
        columns = self._data.shape[1] - self._data.shape[1] % grid_size
        data = self._data[:rows, :columns]
        return np.array([np.split(splitted, grid_size, axis=1) for splitted in np.split(data, grid_size)])

    def _bezier_curves(self, num_of_curves):
        width = self._curve_width
        width_buffer = (width - 1) // 2
        n = self._data.shape[0]
        curves = []

        i = 0
        while i < num_of_curves:

            curve = np.zeros(n)

            r0, c0 = np.random.randint(width_buffer, n // 2, size=2)
            r1, c1 = np.random.randint(width_buffer, n - width_buffer, size=2)
            r2, c2 = np.random.randint(n // 2, n - width_buffer, size=2)
            weight = 2
            rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, weight, self._data.shape)

            if len(rr) < n:
                continue

            if i % 10 == 0:
                print(i)

            rr, cc = rr[len(rr) - n:], cc[len(cc) - n:]

            if np.random.binomial(1, p=0.5):
                cc = n - cc

            for j, (row, column) in enumerate(zip(rr, cc)):
                rr_d, cc_d = disk((row, column), width_buffer)
                curve[j] = np.nanmean(self._data[rr_d, cc_d])

            curves.append(curve)
            i += 1

        curves_noise = self._noise_std / np.sqrt(len(rr_d))

        return np.array(curves), curves_noise

    def _create_curves(self, num, split=1, concat=1, high_power_selection_factor=1, strategy='random lines'):

        if strategy == 'rows':
            jump = self._data.shape[0] // num
            curves = self._rows_curves(jump)

        if strategy == 'random lines':
            curves = self._random_lines_curves(high_power_selection_factor * num)
            curves_noise = self._noise_std / np.sqrt(self._num_of_curves)

        if strategy == 'bezier':
            curves, curves_noise = self._bezier_curves(high_power_selection_factor * num)

        curves_powers = [
            estimate_signal_power(curve, self._noise_std, self._noise_mean, SignalPowerEstimator.FirstMoment)
            for curve in curves]
        curves = np.array(curves)[np.argsort(curves_powers)[-num * concat:]]
        curves = np.array([np.concatenate(x) for x in np.split(curves, num)])

        if split > 1:
            sub_curve_length = self._data.shape[0] // split
            curves = curves[:, :sub_curve_length * split]
            curves = np.array(np.split(curves, split, axis=1)).reshape(-1, sub_curve_length)

        return curves, curves_noise

    def estimate(self):
        fixed_length_options = (np.int32(self._length_options * self._cut_fix_factor))

        likelihoods = np.zeros(len(fixed_length_options))
        powers = np.zeros(len(fixed_length_options))
        for i, length in enumerate(fixed_length_options):
            likelihoods[i], powers[i] = max_argmax_1d_case(self._curves,
                                                           self._signal_filter_gen(length, 1),
                                                           self._fixed_num_of_occurrences,
                                                           self._curves_noise)
            logger.info(
                f'For length {self._length_options[i]} matched power is {powers[i]}, Likelihood={likelihoods[i]}')

        most_likely_length = self._length_options[np.nanargmax(likelihoods)]
        most_likely_power = powers[np.nanargmax(likelihoods)]
        return likelihoods, most_likely_length, most_likely_power
