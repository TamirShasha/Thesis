import numpy as np
from skimage.draw import line

from src.algorithms.length_estimator_1d import LengthExtractor1D
from src.algorithms.multiple_lengths_estimator_1d import MultipleLengthsEstimator1D, CircleCutsDistribution, \
    Ellipse1t2CutsDistribution
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator
from src.utils.logger import logger


class LengthEstimator2DCurvesMethod:

    def __init__(self,
                 data,
                 length_options,
                 signal_filter_gen,
                 signal_area_fraction_boundaries,
                 signal_num_of_occurrences_boundaries,
                 noise_mean,
                 noise_std,
                 signal_power_estimator_method,
                 exp_attr,
                 num_of_power_options=10,
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen
        self._signal_area_fraction_boundaries = signal_area_fraction_boundaries
        self._signal_num_of_occurrences_boundaries = signal_num_of_occurrences_boundaries
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._exp_attr = exp_attr
        self._num_of_power_options = num_of_power_options
        self._n = self._data.shape[0]

        self._num_of_curves = 3 * int(np.log(np.max(self._data.shape)))
        self._curves = self._create_curves(num=self._num_of_curves,
                                           high_power_selection_factor=100,
                                           concat=1)
        logger.info(f'data length: {self._curves.shape[0]} x {self._curves.shape[1]}')
        # print(f'data length: {self._curves.shape[0]} x {self._curves.shape[1]}')

        self._signal_power = estimate_signal_power(self._data, self._noise_std, self._noise_mean,
                                                   method=self._signal_power_estimator_method)
        logger.debug(f'Full signal power: {self._signal_power}')

        self._avg_signal_power_options = self._estimate_avg_signal_power_options()
        logger.info(f'Avg signal power options: {self._avg_signal_power_options}\n')

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

    def _estimate_avg_signal_power_options(self):
        self._signal_instance_power_options_estimation = np.zeros_like(self._num_of_power_options)
        f_min, f_max = self._signal_area_fraction_boundaries
        area_coverage = np.round(np.linspace(f_min, f_max, self._num_of_power_options), 2)
        power_options = np.round(self._signal_power / (np.prod(self._data.shape) * area_coverage), 2)
        return power_options

    def _estimate_likelihood_of_distribution(self, signals_distributions, sep=0):
        logger.debug(f'Will average over {self._num_of_curves} runs (curves)')

        data = self._curves
        best_lengths = []
        sum_likelihoods = np.zeros_like(self._length_options)
        for t in range(self._num_of_curves):
            logger.debug(f'At iteration {t + 1}')
            likelihoods, d = MultipleLengthsEstimator1D(data=data[t],
                                                        length_distribution_options=signals_distributions,
                                                        noise_std=self._noise_std,
                                                        signal_separation=sep).estimate()
            sum_likelihoods = sum_likelihoods + likelihoods
            curr_best_length = self._length_options[np.argmax(sum_likelihoods / (t + 1))]
            best_lengths.append(curr_best_length)

        likelihoods = sum_likelihoods / self._num_of_curves

        return likelihoods, best_lengths

    def _estimate_likelihood_for_1d(self, signal_avg_power):
        logger.debug(f'Will average over {self._num_of_curves} runs (curves)')

        filter_gen = lambda l: self._signal_filter_gen(l, signal_avg_power)

        best_lengths = []
        sum_likelihoods = np.zeros_like(self._length_options)
        for t in range(self._num_of_curves):
            logger.debug(f'At iteration {t}')
            likelihoods, best_len = LengthExtractor1D(data=self._curves[t],
                                                      length_options=self._length_options,
                                                      signal_filter_gen=filter_gen,
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

    def estimate(self, distributions=('1d')):
        likelihoods = dict()

        if 'circle' in distributions:
            logger.info(f'Running for circle distribution')

            likelihoods_circle = np.zeros(shape=(len(self._avg_signal_power_options), len(self._length_options)))
            for i, power in enumerate(self._avg_signal_power_options):
                logger.debug(f'Calculating likelihood for power={power}')
                filter_gen = lambda l: self._signal_filter_gen(l, power)
                signals_distributions = [CircleCutsDistribution(length=l, filter_gen=filter_gen)
                                         for l in self._length_options]
                circle_likelihoods, circle_best_ds = self._estimate_likelihood_of_distribution(signals_distributions, 0)
                likelihoods_circle[i] = circle_likelihoods

            likelihoods['circle'] = np.max(likelihoods_circle, 0)

        if 'ellipse12' in distributions:
            logger.info(f'Running for ellipse 1:2 distribution')

            likelihoods_ellipse12 = np.zeros(shape=(len(self._avg_signal_power_options), len(self._length_options)))
            for i, power in enumerate(self._avg_signal_power_options):
                filter_gen = lambda l: self._signal_filter_gen(l, power)
                signals_distributions = [Ellipse1t2CutsDistribution(length=l, filter_gen=filter_gen)
                                         for l in self._length_options]
                ellipse_likelihoods, best_l = self._estimate_likelihood_of_distribution(signals_distributions, 0)
                likelihoods_ellipse12[i] = ellipse_likelihoods

            likelihoods['ellipse12'] = np.max(likelihoods_ellipse12, 0)

        if '1d' in distributions:
            logger.info('Running for 1d')

            likelihoods_1d = np.zeros(shape=(len(self._avg_signal_power_options), len(self._length_options)))
            for i, power in enumerate(self._avg_signal_power_options):
                logger.info(f'Running for 1d, power={power}')
                one_d_likelihoods, one_d_best_lengths = self._estimate_likelihood_for_1d(power)
                likelihoods_1d[i] = one_d_likelihoods

            likelihoods['1d'] = np.max(likelihoods_1d, 0)

        most_likely_length = self._length_options[np.argmax(likelihoods['1d'])]
        return likelihoods, most_likely_length
