import numpy as np
from skimage.draw import line, line_nd

from src.algorithms.length_estimator_1d import LengthEstimator1D
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator
from src.utils.logger import logger
from src.utils.logsumexp import logsumexp


class LengthEstimator2DCurvesMethod:

    def __init__(self,
                 data,
                 length_options,
                 power_options,
                 num_of_occ_estimation,
                 num_of_occ_estimation_mask,
                 signal_filter_gen_1d,
                 noise_mean,
                 noise_std,
                 signal_power_estimator_method,
                 exp_attr,
                 curve_width=31,
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen_1d
        self._power_options = power_options
        self._num_of_occ_estimation = num_of_occ_estimation
        self._tuples_mask = num_of_occ_estimation_mask
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._curve_width = curve_width
        self._logs = logs
        self._exp_attr = exp_attr
        self._n = self._data.shape[0]

        # self._num_of_curves = 200 * int(np.log(np.max(self._data.shape)))
        self._num_of_curves = 30

        self._curves = self._create_curves(num=self._num_of_curves)

        self._cut_fix_factor = 0.7

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

    def _estimate_likelihood_for_1d(self, signal_avg_power, lengths_mask):
        logger.debug(f'Will average over {self._num_of_curves} runs (curves)')

        def filter_gen(length):
            return self._signal_filter_gen(length, signal_avg_power)

        length_options = (np.int32(self._length_options * self._cut_fix_factor))[lengths_mask]

        if length_options.size == 0:
            return np.full_like(self._length_options, fill_value=-np.inf, dtype=float), []

        non_inf_threshold = 0.8
        best_lengths = []
        non_inf_count = np.zeros_like(length_options)
        sum_likelihoods = np.zeros_like(length_options, dtype=float)
        likelihoods = np.zeros(shape=(self._num_of_curves, len(length_options)))
        for t in range(self._num_of_curves):
            logger.debug(f'At iteration {t}')
            likelihoods[t], best_len = LengthEstimator1D(data=self._curves[t],
                                                         length_options=length_options,
                                                         signal_filter_gen=filter_gen,
                                                         noise_mean=self._noise_mean,
                                                         noise_std=self._noise_std / np.sqrt(self._curve_width),
                                                         signal_power_estimator_method=self._signal_power_estimator_method,
                                                         separation=0.3,
                                                         fixed_occurrences=3,
                                                         exp_attr=self._exp_attr,
                                                         logs=self._logs).estimate()

            non_inf_count += np.where(likelihoods[t] == -np.inf, 0, 1)
            sum_likelihoods += np.where(likelihoods[t] == -np.inf, 0, likelihoods[t])
            curr_best_length = length_options[np.argmax(sum_likelihoods / (t + 1))]
            best_lengths.append(curr_best_length)

        # likelihoods = logsumexp(likelihoods, axis=0)
        likelihoods = sum_likelihoods / non_inf_count
        likelihoods[non_inf_count / self._num_of_curves < non_inf_threshold] = -np.inf

        full_likelihoods = np.full_like(self._length_options, fill_value=-np.inf, dtype=float)
        full_likelihoods[lengths_mask] = likelihoods

        fixed_best_lengths = np.array(best_lengths) // self._cut_fix_factor

        return full_likelihoods, fixed_best_lengths

    def estimate(self):
        likelihoods = dict()

        likelihoods_arr = np.zeros(shape=(len(self._power_options), len(self._length_options)))
        for i, power in enumerate(self._power_options):
            logger.info(f'Running for power={power} ({i + 1}/{len(self._power_options)})')
            power_likelihoods, one_d_best_lengths = self._estimate_likelihood_for_1d(power, self._tuples_mask[i])
            likelihoods_arr[i] = power_likelihoods
            likelihoods[f'p_{power}'] = power_likelihoods

        max_row = np.argmax(np.max(likelihoods_arr, axis=1))
        logger.info(f'Power = {self._power_options[max_row]} yielded the maximum likelihood')
        likelihoods['max'] = likelihoods_arr[max_row]
        most_likely_length = self._length_options[np.argmax(likelihoods['max'])]
        return likelihoods, most_likely_length
