import numpy as np
from enum import Enum
import logging
import matplotlib.pyplot as plt

from src.algorithms.filter_estimator_1d import FilterEstimator1D
from src.utils.common_filter_basis import create_filter_basis_1d
from src.algorithms.signal_power_estimator import SignalPowerEstimator as SPE
from src.utils.logger import logger


class SignalPowerEstimator(SPE, Enum):
    Exact = "Exact Power"


class SizeEstimator1D:

    def __init__(self,
                 data,
                 signal_size_by_percentage=None,
                 num_of_instances=40,
                 prior_filter=None,
                 noise_mean=None,
                 noise_std=None,
                 estimate_noise_parameters=True,
                 use_noise_params=True,
                 filter_basis_size=20,
                 particles_margin=0.01,
                 save_statistics=False,
                 log_level=logging.INFO,
                 plots=False,
                 save=False,
                 experiment_dir=None,
                 experiment_attr=None):
        self._data = data
        self._signal_length_by_percentage = signal_size_by_percentage
        self._num_of_instances = num_of_instances
        self._prior_filter = prior_filter
        self._noise_mean = None
        self._noise_std = None
        self._estimate_noise_parameters = estimate_noise_parameters
        self._use_noise_params = use_noise_params
        self._filter_basis_size = filter_basis_size
        self._particles_margin = particles_margin
        self._save_statistics = save_statistics
        self._log_level = log_level
        self._plots = plots
        self._save = save
        self._experiment_dir = experiment_dir
        self._experiment_attr = experiment_attr

        self._data_size = self._data.shape[1]

        if self._use_noise_params:
            self._noise_mean = noise_mean
            self._noise_std = noise_std

        if self._signal_length_by_percentage is None:
            self._signal_length_by_percentage = [4, 6, 8, 10, 12]

        self._signal_size_options = np.array(np.array(self._signal_length_by_percentage) / 100 * self._data_size,
                                             dtype=int)
        logger.info(f'Particle size options are: {self._signal_size_options}')

        logger.setLevel(log_level)

    def estimate(self):
        margin = int(self._particles_margin * self._data_size) // 2
        logger.info(f'Particles margin is {margin * 2} pixels')

        likelihoods = np.zeros(len(self._signal_size_options))
        optimal_coeffs = np.zeros(shape=(len(self._signal_size_options), self._filter_basis_size))
        for i, size in enumerate(self._signal_size_options):
            logger.info(f'Estimating likelihood for size={size}')

            filter_basis = create_filter_basis_1d(size, self._filter_basis_size)
            filter_estimator = FilterEstimator1D(unnormalized_data=self._data,
                                                 unnormalized_filter_basis=filter_basis,
                                                 num_of_instances=self._num_of_instances,
                                                 noise_std=self._noise_std,
                                                 noise_mean=self._noise_mean)

            likelihoods[i], optimal_coeffs[i] = filter_estimator.estimate()

            logger.info(
                f'For length {self._signal_size_options[i]}, Likelihood={likelihoods[i]}')

        most_likely_index = np.nanargmax(likelihoods)
        filter_basis = create_filter_basis_1d(self._signal_size_options[most_likely_index], self._filter_basis_size)
        est_signals = filter_basis.T.dot(optimal_coeffs[most_likely_index])

        return likelihoods, optimal_coeffs, est_signals
