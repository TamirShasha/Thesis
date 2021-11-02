import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import os

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
                 length_options,
                 fixed_num_of_occurrences=30,
                 noise_mean=0,
                 noise_std=1,
                 filter_basis_size=20,
                 particles_margin=0.01,
                 logs=True,
                 plots=False,
                 save=False,
                 experiment_dir=None):
        self._data = data
        self._length_options = length_options
        self._fixed_num_of_occurrences = fixed_num_of_occurrences
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._filter_basis_size = filter_basis_size
        self._particles_margin = particles_margin
        self._logs = logs
        self._plots = plots
        self._save = save
        self._experiment_dir = experiment_dir
        self._n = self._data.shape[0]

        self.log_prob_all_noise = utils.log_prob_all_is_noise(self._data, self._noise_std)

    def _save_results(self, likelihoods, optimal_coeffs):
        parameters = {
            "length_options": self._length_options,
            "fixed_num_of_occurrences": self._fixed_num_of_occurrences,
            "noise_mean": self._noise_mean,
            "noise_std": self._noise_std,
            "filter_basis_size": self._filter_basis_size
        }

        results = {
            "likelihoods": likelihoods,
            "optimal_coeffs": optimal_coeffs
        }

    def estimate(self):

        margin = int(self._particles_margin * self._data.shape[0]) // 2
        logger.info(f'Particles margin is {margin * 2} pixels')

        likelihoods = np.zeros(len(self._length_options))
        optimal_coeffs = np.zeros(shape=(len(self._length_options), self._filter_basis_size))
        for i, length in enumerate(self._length_options):

            logger.info(f'Estimating likelihood for size={length}')

            filter_basis = create_filter_basis(length, self._filter_basis_size)
            filter_basis = np.array(
                [np.pad(x, ((margin, margin), (margin, margin)), 'constant', constant_values=((0, 0), (0, 0)))
                 for x in filter_basis])

            filter_estimator = FilterEstimator2D(self._data,
                                                 filter_basis,
                                                 self._fixed_num_of_occurrences,
                                                 self._noise_std)

            likelihoods[i], optimal_coeffs[i] = filter_estimator.estimate()
            est_signal = filter_basis.T.dot(optimal_coeffs[i])

            if self._save:
                plt.imshow(est_signal, cmap='gray')
                plt.colorbar()
                fig_path = os.path.join(self._experiment_dir, f'_{length}.png')
                plt.savefig(fname=fig_path)
                plt.close()

            logger.info(
                f'For length {self._length_options[i]}, Likelihood={likelihoods[i]}')

        most_likely_index = np.nanargmax(likelihoods)
        filter_basis = create_filter_basis(self._length_options[most_likely_index], self._filter_basis_size)
        est_signals = filter_basis.T.dot(optimal_coeffs[most_likely_index])

        # if self._save:
        #     self._save_results(likelihoods, optimal_coeffs)

        return likelihoods, optimal_coeffs, est_signals
