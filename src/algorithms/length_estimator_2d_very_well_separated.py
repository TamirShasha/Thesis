import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import os
import logging

import src.algorithms.utils as utils
from src.algorithms.utils import cryo_downsample
from src.algorithms.filter_estimator_2d import FilterEstimator2D, create_filter_basis
from src.algorithms.signal_power_estimator import SignalPowerEstimator as SPE
from src.utils.logger import logger


class SignalPowerEstimator(SPE, Enum):
    Exact = "Exact Power"


class LengthEstimator2DVeryWellSeparated:

    def __init__(self,
                 data,
                 signal_length_by_percentage=None,
                 num_of_instances_range=(1, 100),
                 noise_mean=0,
                 noise_std=1,
                 estimate_noise_parameters=True,
                 filter_basis_size=20,
                 particles_margin=0.01,
                 estimate_locations_and_num_of_instances=False,
                 log_level=logging.INFO,
                 plots=False,
                 save=False,
                 experiment_dir=None,
                 experiment_attr=None):
        self._data = data
        self._signal_length_by_percentage = signal_length_by_percentage
        self._num_of_instances_range = num_of_instances_range
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._estimate_noise_parameters = estimate_noise_parameters
        self._filter_basis_size = filter_basis_size
        self._particles_margin = particles_margin
        self._estimate_locations_and_num_of_instances = estimate_locations_and_num_of_instances
        self._log_level = log_level
        self._plots = plots
        self._save = save
        self._experiment_dir = experiment_dir
        self._experiment_attr = experiment_attr

        self._data_size = self._data.shape[0]

        if self._signal_length_by_percentage is None:
            self._signal_length_by_percentage = [4, 6, 8, 10, 12]

        self._signal_size_options = np.array(np.array(self._signal_length_by_percentage) / 100 * self._data_size,
                                             dtype=int)
        logger.info(f'Particle size options are: {self._signal_size_options}')

        logger.setLevel(log_level)

    def estimate(self):
        margin = int(self._particles_margin * self._data.shape[0]) // 2
        logger.info(f'Particles margin is {margin * 2} pixels')

        likelihoods = np.zeros(len(self._signal_size_options))
        optimal_coeffs = np.zeros(shape=(len(self._signal_size_options), self._filter_basis_size))
        for i, length in enumerate(self._signal_size_options):
            logger.info(f'Estimating likelihood for size={length}')

            filter_basis = create_filter_basis(length, self._filter_basis_size)
            filter_estimator = FilterEstimator2D(unnormalized_data=self._data,
                                                 unnormalized_filter_basis=filter_basis,
                                                 num_of_instances_range=self._num_of_instances_range,
                                                 noise_std=self._noise_std,
                                                 noise_mean=self._noise_mean,
                                                 estimate_noise_parameters=self._estimate_noise_parameters,
                                                 signal_margin=margin,
                                                 estimate_locations_and_num_of_instances=self._estimate_locations_and_num_of_instances,
                                                 experiment_dir=self._experiment_dir,
                                                 experiment_attr=self._experiment_attr,
                                                 plots=self._plots,
                                                 log_level=self._log_level)

            likelihoods[i], optimal_coeffs[i] = filter_estimator.estimate()
            logger.info(
                f'For length {self._signal_size_options[i]}, Likelihood={likelihoods[i]}')

        most_likely_index = np.nanargmax(likelihoods)
        filter_basis = create_filter_basis(self._signal_size_options[most_likely_index], self._filter_basis_size)
        est_signals = filter_basis.T.dot(optimal_coeffs[most_likely_index])

        return likelihoods, optimal_coeffs, est_signals
