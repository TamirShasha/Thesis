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
                 downsample_to_num_of_rows=1000,
                 filter_basis_size=20,
                 logs=True,
                 plots=False,
                 save=False,
                 experiment_dir=None):
        self._data = data
        self._length_options = length_options
        self._fixed_num_of_occurrences = fixed_num_of_occurrences
        self._noise_mean = noise_mean
        # self._noise_mean = np.nanmean(data)
        self._noise_std = noise_std
        # self._noise_std = np.nanstd(data)
        self._filter_basis_size = filter_basis_size
        self._logs = logs
        self._plots = plots
        self._save = save
        self._experiment_dir = experiment_dir

        self._n = self._data.shape[0]
        # if self._n > downsample_to_num_of_rows:
        #     self._downsample_factor = (self._n / downsample_to_num_of_rows)
        #     self._data = cryo_downsample(self._data, (downsample_to_num_of_rows, downsample_to_num_of_rows))
        #     self._noise_std = self._noise_std / self._downsample_factor
        #     self._n = self._data.shape[0]
        #     if self._plots:
        #         plt.imshow(self._data, cmap='gray')
        #         plt.show()
        #     self._length_options = np.array(np.ceil(self._length_options / self._downsample_factor), dtype=int)
        #     logger.info(f'Length options after downsample: {self._length_options}')

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

        likelihoods = np.zeros(len(self._length_options))
        optimal_coeffs = np.zeros(shape=(len(self._length_options), self._filter_basis_size))
        for i, length in enumerate(self._length_options):

            filter_basis = create_filter_basis(length, self._filter_basis_size)
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
