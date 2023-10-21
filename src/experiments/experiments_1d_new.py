import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR
import logging
from datetime import datetime
import pathlib
import pickle

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
        if self._plot:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(self._data[0][:5000], label='noisy data')
            ax.plot(self._data_simulator.clean_data[:5000], label='underlying signals')
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.xaxis.set_visible(False)
            plt.tight_layout()
            plt.show()

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
            "most_likely_index": np.nanargmax(likelihoods),
            "most_likely_size": self._sizes_options[np.nanargmax(likelihoods)]
        }

        self.save_and_plot_all_figures()
        # self.save_and_plot_reduced()

        return self._results

    def save_and_plot_original(self):

        likelihoods = self._results['likelihoods']
        most_likely_index = np.nanargmax(likelihoods)

        if self._save:
            data_to_save = {
                **self._results,
                **{
                    "signal_size": self._signal_length,
                    "sizes_options": self._sizes_options,
                    "number_of_instances": self._num_of_instances,
                    "noise_mean": self._noise_mean,
                    "noise_std": self._noise_std,
                    "most_likely_index": most_likely_index
                }
            }
            pickle.dump(data_to_save, open(os.path.join(self.experiment_dir, 'data.pkl'), 'wb'))

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
        axs[1, 1].plot(self._sizes_options, self._results['likelihoods'], 'o-')
        axs[1, 1].set_title('Likelihoods')

        if self._save:
            fig_path = os.path.join(self.experiment_dir, f'{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()

    def save_and_plot_all_figures(self):

        font = {'size': 12}
        import matplotlib
        matplotlib.rc('font', **font)

        likelihoods = self._results['likelihoods']
        most_likely_index = np.nanargmax(likelihoods)

        if self._save:
            data_to_save = {
                **self._results,
                **{
                    "data": self._data[0],
                    "clean_data": self._data_simulator.clean_data,
                    "signal_size": self._signal_length,
                    "sizes_options": self._sizes_options,
                    "number_of_instances": self._num_of_instances,
                    "noise_mean": self._noise_mean,
                    "noise_std": self._noise_std,
                    "most_likely_index": most_likely_index
                }
            }
            pickle.dump(data_to_save, open(os.path.join(self.experiment_dir, 'data.pkl'), 'wb'))

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.yaxis.set_visible(False)

        ax1.plot(self._data[0][:5000])
        # ax1.set_title('Noisy Data')
        ax1.plot(self._data_simulator.clean_data[:5000])
        # ax1.legend(loc='upper right')
        pad = self._signal_length // 10
        ax2.plot(
            np.pad(self._data_simulator.unmarginized_signal_gen(self._signal_length), (pad, pad), 'constant',
                   constant_values=(0, 0)),
            label='true')
        ax2.plot(np.pad(self._results['estimated_signal'], (pad, pad), 'constant',
                        constant_values=(0, 0)), label='estimated', linestyle='--')
        ax2.legend(loc='lower center')
        # ax2.set_title('True Signal And Estimated Signal')
        ax3.plot(self._sizes_options, self._results['likelihoods'], 'o-')
        ax3.axvline(x=self._signal_length, label='signal true size', color='black', linestyle='--')
        ax3.scatter([self._results['most_likely_size']],
                    [self._results['likelihoods'][self._results['most_likely_index']]],
                    color='red', marker='*', s=300, label='signal estimated size')
        # ax3.set_title('Likelihood')
        ax3.legend(loc='lower right')

        plt.tight_layout()

        if self._save:
            fig_path = os.path.join(self.experiment_dir, f'{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()

    def save_and_plot_reduced(self):

        font = {'size': 12}
        import matplotlib
        matplotlib.rc('font', **font)

        likelihoods = self._results['likelihoods']
        most_likely_index = np.nanargmax(likelihoods)

        if self._save:
            data_to_save = {
                **self._results,
                **{
                    "signal_size": self._signal_length,
                    "sizes_options": self._sizes_options,
                    "number_of_instances": self._num_of_instances,
                    "noise_mean": self._noise_mean,
                    "noise_std": self._noise_std,
                    "most_likely_index": most_likely_index
                }
            }
            pickle.dump(data_to_save, open(os.path.join(self.experiment_dir, 'data.pkl'), 'wb'))

        fig = plt.figure(figsize=(12, 3))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[:2])
        ax1.xaxis.set_visible(False)
        ax2 = fig.add_subplot(gs[2])

        ax1.plot(self._data[0][:5000])
        ax1.plot(self._data_simulator.clean_data[:5000])
        if self._noise_std == 0.1:
            ax1.set_ylim(-8, 8)
        pad = self._signal_length // 10
        ax2.yaxis.set_visible(False)
        ax2.plot(self._sizes_options, self._results['likelihoods'], 'o-')
        ax2.axvline(x=self._signal_length, label='signal true size', color='black', linestyle='--')
        ax2.scatter([self._results['most_likely_size']],
                    [self._results['likelihoods'][self._results['most_likely_index']]],
                    color='red', marker='*', s=300)

        plt.tight_layout()

        if self._save:
            fig_path = os.path.join(self.experiment_dir, f'{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()

    def save_and_plot(self):

        font = {'size': 16}
        import matplotlib
        matplotlib.rc('font', **font)

        likelihoods = self._results['likelihoods']
        most_likely_index = np.nanargmax(likelihoods)

        if self._save:
            data_to_save = {
                **self._results,
                **{
                    "signal_size": self._signal_length,
                    "sizes_options": self._sizes_options,
                    "number_of_instances": self._num_of_instances,
                    "noise_mean": self._noise_mean,
                    "noise_std": self._noise_std,
                    "most_likely_index": most_likely_index
                }
            }
            pickle.dump(data_to_save, open(os.path.join(self.experiment_dir, 'data.pkl'), 'wb'))

        # fig, axs = plt.subplots(2, 2, figsize=(12, 6))
        fig = plt.figure(figsize=(16, 4))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.xaxis.set_visible(False)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.yaxis.set_visible(False)
        # ax3 = fig.add_subplot(gs[1, 1])

        ax1.plot(self._data[0][:5000], label='noisy data')
        ax1.set_title('Noisy Data')
        ax1.plot(self._data_simulator.clean_data[:5000], label='underlying signals')
        ax1.legend(loc='upper right')

        ax2.plot(self._sizes_options, self._results['likelihoods'], 'o-')
        ax2.axvline(x=self._signal_length, label='signal true size', color='black', linestyle='--')
        ax2.scatter([self._results['most_likely_size']],
                    [self._results['likelihoods'][self._results['most_likely_index']]],
                    color='red', marker='*', s=300, label='signal estimated size')
        ax2.set_title('Likelihood')
        ax2.legend(loc='lower right')

        plt.tight_layout()

        if self._save:
            fig_path = os.path.join(self.experiment_dir, f'{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()


np.random.seed(505)

arbitrary_signal = lambda d: 1 - 2 * np.square(np.linspace(0, 1, d) - 0.5 + 0.2 * np.sin(10 * np.linspace(0, 1, d)))
pulses = lambda d: np.ones(d)
paraboly = lambda d: 1 - 2 * np.square(np.linspace(0, 1, d) - 0.5)


def run_experiment_pulses(noise_std=1.0, random_seed=503):
    np.random.seed(random_seed)
    sim_data = DataSimulator1D(size=50000,
                               signal_size=150,
                               # signal_fraction=0.1,
                               signal_margin=0.0001,
                               num_of_instances=80,
                               signal_gen=pulses,
                               noise_std=noise_std,
                               noise_mean=0)

    Experiment1D(
        name=f"experiment_pulses_1d_{str(noise_std).replace('.', '_')}",
        simulator=sim_data,
        signal_length_by_percentage=np.arange(0.1, 2, 0.1) * sim_data.signal_size / sim_data.size * 100,
        # signal_length_by_percentage=np.array([0]) * sim_data.signal_size / sim_data.size * 100,
        num_of_instances=sim_data.num_of_instances,
        use_noise_params=True,
        estimate_noise=False,
        filter_basis_size=1,
        save_statistics=False,
        # prior_filter=lambd

        # a d: np.ones(d),
        particles_margin=0,
        plot=False,
        save=True,
        log_level=logging.INFO
    ).run()


def run_experiment_arbitrary(noise_std=1, random_seed=503, signal_gen=paraboly, basis_size=8):
    np.random.seed(random_seed)
    sim_data = DataSimulator1D(size=50000,
                               signal_size=100,
                               # signal_fraction=0.1,
                               signal_margin=0.0001,
                               num_of_instances=150,
                               signal_gen=signal_gen,
                               noise_std=noise_std,
                               noise_mean=0)

    Experiment1D(
        name=f"experiment_{signal_gen.__name__}_1d_{str(noise_std).replace('.', '_')}",
        simulator=sim_data,
        signal_length_by_percentage=np.arange(0.1, 2, 0.1) * sim_data.signal_size / sim_data.size * 100,
        # signal_length_by_percentage=np.array([0]) * sim_data.signal_size / sim_data.size * 100,
        num_of_instances=sim_data.num_of_instances,
        use_noise_params=True,
        estimate_noise=False,
        filter_basis_size=basis_size,
        save_statistics=False,
        # prior_filter=lambd

        # a d: np.ones(d),
        particles_margin=0,
        plot=True,
        save=True,
        log_level=logging.INFO
    ).run()


def __main__():
    # run_experiment_pulses(noise_std=0.1, random_seed=503)
    # run_experiment_pulses(noise_std=3, random_seed=503)
    # run_experiment_pulses(noise_std=10, random_seed=503)
    # run_experiment_pulses(noise_std=50, random_seed=503)
    # run_experiment_arbitrary(noise_std=1, signal_gen=paraboly, basis_size=3)
    run_experiment_arbitrary(noise_std=1, signal_gen=arbitrary_signal)

    # noise_std = [1]
    # noise_std = [0.1, 5, 10, 50]
    # noise_std = [10, 20]

    # for noise_std in [0.1, 5, 10, 50]:
    #     for random_seed in np.arange(505, 506):
    #         np.random.seed()
    #         sim_data = DataSimulator1D(size=300000,
    #                                    signal_size=150,
    #                                    # signal_fraction=0.1,
    #                                    signal_margin=0.0001,
    #                                    num_of_instances=500,
    #                                    signal_gen=pulses,
    #                                    noise_std=noise_std,
    #                                    noise_mean=0)
    #
    #         Experiment1D(
    #             name=f"expy_{noise_std}",
    #             simulator=sim_data,
    #             signal_length_by_percentage=np.arange(0.1, 2, 0.1) * sim_data.signal_size / sim_data.size * 100,
    #             # signal_length_by_percentage=np.array([0]) * sim_data.signal_size / sim_data.size * 100,
    #             num_of_instances=sim_data.num_of_instances,
    #             use_noise_params=True,
    #             estimate_noise=False,
    #             filter_basis_size=1,
    #             save_statistics=False,
    #             # prior_filter=lambd
    #             # a d: np.ones(d),
    #             particles_margin=0,
    #
    #             plot=False,
    #             save=True,
    #             log_level=logging.INFOs
    #         ).run()

    # for noise_std in [0.1]:
    #
    # np.random.seed(500)
    # sim_data = DataSimulator1D(size=10000,
    #                            signal_size=50,
    #                            # signal_fraction=0.1,
    #                            signal_margin=0.0001,
    #                            num_of_instances=50,
    #                            signal_gen=pulses,
    #                            noise_std=0.1,
    #                            noise_mean=0)
    #
    # Experiment1D(
    #     name=f"expy_{0.1}",
    #     simulator=sim_data,
    #     signal_length_by_percentage=np.arange(0.1, 2, 0.1) * sim_data.signal_size / sim_data.size * 100,
    #     # signal_length_by_percentage=np.array([0]) * sim_data.signal_size / sim_data.size * 100,
    #     num_of_instances=sim_data.num_of_instances,
    #     use_noise_params=True,
    #     estimate_noise=False,
    #     filter_basis_size=1,
    #     save_statistics=False,
    #     # prior_filter=lambd
    #     # a d: np.ones(d),
    #     particles_margin=0,
    #     plot=False,
    #     save=True,
    #     log_level=logging.INFO
    # ).run()


if __name__ == '__main__':
    __main__()
