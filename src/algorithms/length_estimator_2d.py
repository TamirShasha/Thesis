import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
from enum import Enum

from src.algorithms.length_estimator_2d_curves_method import LengthEstimator2DCurvesMethod
from src.algorithms.length_estimator_2d_sep_method import LengthEstimator2DSeparationMethod
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator
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
        self._exp_attr = exp_attr
        self._num_of_power_options = num_of_power_options
        self._n = self._data.shape[0]

        self._signal_power = estimate_signal_power(self._data, self._noise_std, self._noise_mean,
                                                   method=self._signal_power_estimator_method)

        if self._signal_power < 0:
            self._data = -self._data
            self._signal_power = estimate_signal_power(self._data, self._noise_std, self._noise_mean,
                                                       method=self._signal_power_estimator_method)
        logger.info(f'Full signal power: {self._signal_power}')

        self._avg_signal_power_options = self._estimate_avg_signal_power_options()
        logger.info(f'Avg signal power options: {self._avg_signal_power_options}\n')

        self._dpk, self._dpk_mask = self._generate_dpk_tuples()

        logger.info('Done preprocessing .')

    def _estimate_avg_signal_power_options(self):
        f_min, f_max = self._signal_area_fraction_boundaries
        area_coverage = np.round(np.linspace(f_min, f_max, self._num_of_power_options), 2)
        power_options = np.round(self._signal_power / (np.prod(self._data.shape) * area_coverage), 4)
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

        mask = np.zeros(shape=(len(self._length_options), self._num_of_power_options))
        dpk = np.zeros(shape=(len(self._length_options), self._num_of_power_options), dtype=int)
        for i, signal_length in enumerate(self._length_options):
            for j, signal_power in enumerate(self._avg_signal_power_options):
                expected_num_of_occurrences = self._estimate_num_of_signal_occurrences(signal_length, signal_power)
                dpk[i, j] = expected_num_of_occurrences
                is_possible = True
                if expected_num_of_occurrences < min_occurrences or expected_num_of_occurrences > max_occurrences:
                    is_possible = False

                mask[i, j] = is_possible

        fig, ax = plt.subplots()
        ax.imshow(dpk.T, aspect='auto')

        ax.set_xticks(np.arange(len(self._length_options)))
        ax.set_yticks(np.arange(len(self._avg_signal_power_options)))
        ax.set_xticklabels(self._length_options)
        ax.set_yticklabels(self._avg_signal_power_options)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        for i in range(len(self._length_options)):
            for j in range(len(self._avg_signal_power_options)):
                color = "w" if mask[i, j] else "r"
                ax.text(i, j, dpk[i, j],
                        ha="center", va="center", color=color)

        ax.set_title("Power & Length combinations")
        fig.tight_layout()
        plt.show()

        return dpk, mask

    def estimate(self):
        if self._estimation_method == EstimationMethod.Curves:
            logger.info(f'Estimating signal length using well separation method')
            length_estimator = \
                LengthEstimator2DCurvesMethod(data=self._data,
                                              length_options=self._length_options,
                                              power_options=self._avg_signal_power_options,
                                              num_of_occ_estimation=self._dpk,
                                              tuples_mask=self._dpk_mask,
                                              signal_filter_gen_1d=self._signal_filter_gen_1d,
                                              noise_mean=self._noise_mean,
                                              noise_std=self._noise_std,
                                              signal_power_estimator_method=self._signal_power_estimator_method,
                                              exp_attr=self._exp_attr,
                                              logs=self._logs)
        else:
            logger.info(f'Estimating signal length using curves method')
            # self._length_estimator = LengthEstimator2D(data=self._data,
            #                                            length_options=self._signal_length_options,
            #                                            signal_area_fraction_boundaries=signal_area_coverage_boundaries,
            #                                            signal_num_of_occurrences_boundaries=signal_num_of_occurrences_boundaries,
            #                                            num_of_power_options=num_of_power_options,
            #                                            signal_filter_gen_1d=signal_1d_filter_gen,
            #                                            signal_filter_gen_2d=signal_2d_filter_gen,
            #                                            noise_mean=self._noise_mean,
            #                                            noise_std=self._noise_std,
            #                                            signal_power_estimator_method=signal_power_estimator_method,
            #                                            exp_attr=self._exp_attr,
            #                                            logs=self._logs)

        likelihoods, most_likely_length = length_estimator.estimate()
        return likelihoods, most_likely_length
