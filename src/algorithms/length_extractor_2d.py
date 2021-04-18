import numpy as np

from src.algorithms.length_extractor_1d import LengthExtractor1D


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

    def _create_curvature(self, jump=None):
        if jump is None:
            jump = int(self._y.shape[0] / 20)

        mat = self._y
        curve_rows = mat[np.arange(0, mat.shape[0], jump)]
        curve = np.concatenate(curve_rows)
        return curve

    def extract(self):
        # scan over the data to create 1d data
        y_1d = self._create_curvature()

        print(f'data 1d length is {y_1d.shape[0]}')

        # create length extractor
        length_extractor_1d = LengthExtractor1D(y_1d,
                                                self._length_options,
                                                self._signal_filter_gen,
                                                self._noise_mean,
                                                self._noise_std,
                                                self._signal_power_estimator_method,
                                                self._exp_attr,
                                                self._logs)
        likelihoods, d = length_extractor_1d.extract()
        return likelihoods, d
