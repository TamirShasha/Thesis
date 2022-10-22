import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR
import logging
from datetime import datetime
import pathlib

from src.experiments.data_simulator_1d import DataSimulator1D
from src.utils.logger import logger
from src.algorithms.size_estimator_1d import SizeEstimator1D

# np.random.seed(501)
logger.setLevel(logging.INFO)


class Experiment1D:

    def __init__(self,
                 name=str(time.time()),
                 simulator=None,
                 signal_length_by_percentage=None,
                 filter_basis_size=20,
                 num_of_instances=40,
                 prior_filter=None,
                 particles_margin=0.01,
                 estimate_noise=False,
                 use_noise_params=True,
                 save_statistics=False,
                 plot=True,
                 save=True,
                 save_dir=os.path.join(ROOT_DIR, f'src/experiments/plots/'),
                 log_level=logging.INFO):
        self._name = name  # Experiment name
        self._data_simulator = simulator
        self._filter_basis_size = filter_basis_size
        self._num_of_instances = num_of_instances
        self._prior_filter = prior_filter
        self._estimate_noise = estimate_noise
        self._use_noise_params = use_noise_params
        self._particles_margin = particles_margin
        self._save_statistics = save_statistics

        self._plot = plot
        self._save = save
        self._save_dir = save_dir
        self._log_level = log_level
        self._results = {}
        self.experiment_attr = {}

        curr_date = str(datetime.now().strftime("%d-%m-%Y"))
        curr_time = str(datetime.now().strftime("%H-%M-%S"))
        self.experiment_dir = os.path.join(self._save_dir, curr_date, curr_time)

        if simulator is None:
            simulator = DataSimulator1D()
        self._data = np.array([simulator.simulate()])

        logger.info(f'Simulating data, number of occurrences is {simulator.num_of_instances}')

        self._noise_std = simulator.noise_std
        self._noise_mean = simulator.noise_mean

        self._signal_length = simulator.signal_size

        self.experiment_attr['clean_data'] = simulator.clean_data
        self.experiment_dir += f'_size_{simulator.signal_size}_std_{self._noise_std}'

        self._data_size = self._data.shape[1]

        plt.rcParams["figure.figsize"] = (16, 9)
        # if self._plot:
        #     plt.plot(self._data[0])
        #     plt.show()

        if signal_length_by_percentage is None:
            signal_length_by_percentage = [3, 4, 5, 6, 8, 10]
        self._signal_size_by_percentage = np.array(signal_length_by_percentage)
        self._sizes_options = np.array(self._data_size * self._signal_size_by_percentage // 100, dtype=int)

        if self._save:
            pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        self._estimator = SizeEstimator1D(data=self._data,
                                          signal_size_by_percentage=self._signal_size_by_percentage,
                                          num_of_instances=self._num_of_instances,
                                          prior_filter=self._prior_filter,
                                          noise_mean=self._noise_mean,
                                          noise_std=self._noise_std,
                                          estimate_noise_parameters=self._estimate_noise,
                                          use_noise_params=self._use_noise_params,
                                          filter_basis_size=self._filter_basis_size,
                                          particles_margin=self._particles_margin,
                                          save_statistics=self._save_statistics,
                                          log_level=self._log_level,
                                          plots=self._plot,
                                          save=self._save,
                                          experiment_dir=self.experiment_dir,
                                          experiment_attr=self.experiment_attr)

    def run(self):
        start_time = time.time()
        likelihoods, optimal_coeffs, estimated_signal = self._estimator.estimate()
        end_time = time.time()

        self._results = {
            "likelihoods": likelihoods,
            "optimal_coeffs": optimal_coeffs,
            "estimated_signal": estimated_signal,
            "total_time": end_time - start_time,
            "most_likely_size": self._sizes_options[np.nanargmax(likelihoods)]
        }

        self.save_and_plot()

        return self._results

    def save_and_plot(self):

        likelihoods = self._results['likelihoods']
        most_likely_index = np.nanargmax(likelihoods)

        fig, axs = plt.subplots(2, 2, figsize=(12, 6))
        plt.suptitle(
            f"Size={self._data_size}, Signal Size={self._signal_length}, Num of instances={self._num_of_instances}, "
            f"Noise Mean={self._noise_mean}, Noise STD={self._noise_std} \n"
            f"Most likely size={self._results['most_likely_size']}, Took {'%.3f' % (self._results['total_time'])} Seconds")
        axs[0, 0].plot(self._data[0])
        axs[0, 0].set_title('Noisy Data')
        axs[0, 1].plot(self._data_simulator.clean_data)
        axs[0, 1].set_title('Underlying signals')
        pad = self._signal_length // 10
        axs[1, 0].plot(
            np.pad(self._data_simulator.unmarginized_signal_gen(self._signal_length), (pad, pad), 'constant',
                   constant_values=(0, 0)),
            label='instance')
        axs[1, 0].plot(np.pad(self._results['estimated_signal'], (pad, pad), 'constant',
                              constant_values=(0, 0)), label='estimated', linestyle='--')
        axs[1, 0].legend()
        axs[1, 0].set_title('Signal instance and estimated')
        axs[1, 1].plot(self._sizes_options, self._results['likelihoods'])
        axs[1, 1].set_title('Likelihoods')

        if self._save:
            fig_path = os.path.join(self.experiment_dir, f'{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()


np.random.seed(504)


def __main__():
    sim_data = DataSimulator1D(size=50000,
                               signal_size=100,
                               signal_fraction=0.1,
                               signal_margin=0.001,
                               # num_of_instances=30,
                               signal_gen=lambda d: np.ones(d),
                               noise_std=.5
                               ,
                               noise_mean=0)

    Experiment1D(
        name=f"expy",
        simulator=sim_data,
        # signal_length_by_percentage=[0.5, 1, 2, 3, 4, 5],
        signal_length_by_percentage=[.1, .2, .3],
        num_of_instances=30,
        use_noise_params=True,
        estimate_noise=False,
        filter_basis_size=1,
        save_statistics=False,
        particles_margin=0,
        plot=True,
        save=False,
        log_level=logging.INFO
    ).run()


if __name__ == '__main__':
    __main__()
