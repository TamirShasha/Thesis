import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from multiprocessing import Pool

from src.algorithms.length_estimator_2d_curves_method import LengthEstimator2DCurvesMethod
from src.algorithms.length_estimator_2d_sep_method import LengthEstimator2DSeparationMethod
from src.algorithms.signal_power_estimator import estimate_signal_power
from src.utils.logger import logger


class EstimationMethod(Enum):
    Curves = 0,
    WellSeparation = 1


class LengthEstimator2D:

    def __init__(self,
                 data,
                 length_options,
                 signal_filter_gen_1d,
                 signal_filter_gen_2d,
                 signal_area_fraction_boundaries,
                 signal_num_of_occurrences_boundaries,
                 noise_mean,
                 noise_std,
                 signal_power_estimator_method,
                 estimation_method,
                 exp_attr,
                 num_of_power_options=10,
                 plot=True,
                 logs=True):
        self._data = data
        self._length_options = length_options
        self._signal_filter_gen_1d = signal_filter_gen_1d
        self._signal_filter_gen_2d = signal_filter_gen_2d
        self._signal_area_fraction_boundaries = signal_area_fraction_boundaries
        self._signal_num_of_occurrences_boundaries = signal_num_of_occurrences_boundaries
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._estimation_method = estimation_method
        self._logs = logs
        self._plot = plot
        self._exp_attr = exp_attr
        self._num_of_power_options = num_of_power_options
        self._n = self._data.shape[0]

        self._signal_power = estimate_signal_power(self._data, self._noise_std, self._noise_mean,
                                                   method=self._signal_power_estimator_method)

        if self._signal_power < 0:
            logger.info('Signal power is negative, flipping data.')
            self._data = -self._data
            self._signal_power = estimate_signal_power(self._data, self._noise_std, self._noise_mean,
                                                       method=self._signal_power_estimator_method)
        logger.info(f'Full signal power: {self._signal_power}')

        self._avg_signal_power_options = self._estimate_avg_signal_power_options()
        self._dpk, self._dpk_mask = self._generate_dpk_tuples()

        # if self._plot:
        #     fig, ax = plt.subplots()
        #     self.generate_dpk_plot_table(ax)
        #     fig.tight_layout()
        #     plt.show()

        logger.info('Done preprocessing .\n')

    def _estimate_avg_signal_power_options(self):
        f_min, f_max = self._signal_area_fraction_boundaries
        area_coverage = np.round(np.linspace(f_min, f_max, self._num_of_power_options), 2)
        power_options = np.round(self._signal_power / (np.prod(self._data.shape) * area_coverage), 4)

        info_msg = 'Avg signal power options (Area coverage):'
        for i, (area, power) in enumerate(zip(area_coverage, power_options)):
            if i % 5 == 0:
                info_msg += '\n'
            info_msg += f'{power} ({int(area * 100)}%)\t'
        logger.info(info_msg)

        # _min = np.min(self._data)
        # _max = np.max(self._data)
        # power_options = np.linspace(.1, .2, self._num_of_power_options)

        return power_options

    def _estimate_single_instance_of_signal_power(self, signal_length, signal_power):
        return np.sum(self._signal_filter_gen_2d(signal_length, signal_power))

    def _estimate_num_of_signal_occurrences(self, signal_length, signal_power):
        single_signal_power = self._estimate_single_instance_of_signal_power(signal_length, signal_power)
        k = int(np.round(self._signal_power / single_signal_power))
        return k

    # d stands for length
    # p stands for power
    # k stands for num of occurrences
    def _generate_dpk_tuples(self):
        min_occurrences, max_occurrences = self._signal_num_of_occurrences_boundaries

        mask = np.zeros(shape=(self._num_of_power_options, len(self._length_options)), dtype=bool)
        dpk = np.zeros(shape=(self._num_of_power_options, len(self._length_options)), dtype=int)
        for i, signal_power in enumerate(self._avg_signal_power_options):
            for j, signal_length in enumerate(self._length_options):
                expected_num_of_occurrences = self._estimate_num_of_signal_occurrences(signal_length, signal_power)
                dpk[i, j] = expected_num_of_occurrences
                is_possible = True
                if expected_num_of_occurrences < min_occurrences or expected_num_of_occurrences > max_occurrences:
                    is_possible = False

                mask[i, j] = is_possible

        return dpk, mask

    def generate_dpk_plot_table(self, ax):
        ax.imshow(self._dpk, aspect='auto')

        ax.set_xticks(np.arange(len(self._length_options)))
        ax.set_yticks(np.arange(len(self._avg_signal_power_options)))
        ax.set_xticklabels(self._length_options)
        ax.set_yticklabels(self._avg_signal_power_options)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        for i in range(len(self._length_options)):
            for j in range(len(self._avg_signal_power_options)):
                color = "w" if self._dpk_mask[j, i] else "r"
                if self._dpk[j, i] >= 1000:
                    font_size = 'xx-small'
                else:
                    font_size = 'x-small'
                ax.text(i, j, self._dpk[j, i],
                        ha="center", va="center", color=color, fontsize=font_size)

        ax.set_title("Power & Length combinations")

    def estimate(self):
        if self._estimation_method == EstimationMethod.Curves:
            logger.info(f'Estimating signal length using Curves method')
            length_estimator = \
                LengthEstimator2DCurvesMethod(data=self._data,
                                              length_options=self._length_options,
                                              power_options=self._avg_signal_power_options,
                                              num_of_occ_estimation=self._dpk,
                                              num_of_occ_estimation_mask=self._dpk_mask,
                                              signal_filter_gen_1d=self._signal_filter_gen_1d,
                                              noise_mean=self._noise_mean,
                                              noise_std=self._noise_std,
                                              signal_power_estimator_method=self._signal_power_estimator_method,
                                              exp_attr=self._exp_attr,
                                              logs=self._logs)
        else:
            logger.info(f'Estimating signal length using Well-Separation method')
            length_estimator = \
                LengthEstimator2DSeparationMethod(data=self._data,
                                                  length_options=self._length_options,
                                                  power_options=self._avg_signal_power_options,
                                                  num_of_occ_estimation=self._dpk,
                                                  num_of_occ_estimation_mask=self._dpk_mask,
                                                  signal_filter_gen=self._signal_filter_gen_2d,
                                                  noise_mean=self._noise_mean,
                                                  noise_std=self._noise_std,
                                                  exp_attr=self._exp_attr,
                                                  logs=self._logs)

        likelihoods, most_likely_length = length_estimator.estimate()
        return likelihoods, most_likely_length
