import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR
from enum import Enum

from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.algorithms.length_estimator_1d import SignalPowerEstimator
from src.algorithms.length_estimator_2d_sep_method import LengthEstimator2DSeparationMethod
from src.algorithms.length_estimator_2d_curves_method import LengthEstimator2DCurvesMethod
from src.experiments.micrograph import Micrograph, MICROGRAPHS


class EstimationMethod(Enum):
    Curves = 0,
    WellSeparation = 1


class Experiment2D:

    def __init__(self,
                 name=str(time.time()),
                 mrc: Micrograph = None,
                 simulator=DataSimulator2D(),
                 signal_2d_filter_gen=lambda d, p=1: Shapes2D.disk(d, p),
                 signal_1d_filter_gen=lambda d, p=1: np.full(d, p),
                 signal_power_estimator_method=SignalPowerEstimator.Exact,
                 length_options=None,
                 estimation_method: EstimationMethod = EstimationMethod.WellSeparation,
                 plot=True,
                 save=True,
                 logs=True):
        self._name = name  # Experiment name
        self._signal_power_estimator_method = signal_power_estimator_method

        self._plot = plot
        self._save = save
        self._logs = logs
        self._results = {}

        if mrc is None:
            print(f'Simulating data, number of occurrences is {simulator.occurrences}')
            self._data = simulator.simulate()
            self._num_of_occurrences = simulator.occurrences
            self._noise_std = simulator.noise_std
            self._noise_mean = simulator.noise_mean
            self._signal_length = simulator.signal_length

        else:
            print(f'Loading given micrograph from {mrc.name}')
            self._data = mrc.load_mrc()
            self._num_of_occurrences = mrc.occurrences
            self._noise_std = mrc.noise_std
            self._noise_mean = mrc.noise_mean
            self._signal_length = mrc.signal_length

        self._rows = self._data.shape[0]
        self._columns = self._data.shape[1]

        if length_options is None:
            length_options = np.arange(self._signal_length // 4, int(self._signal_length), 10)
        self._signal_length_options = length_options

        exp_attr = {
            "d": self._signal_length,
            "k": self._num_of_occurrences,
        }

        plt.imshow(self._data, cmap='gray')
        plt.show()

        if estimation_method == EstimationMethod.WellSeparation:
            print(f'Estimating signal length using well separation method')
            self._length_estimator = \
                LengthEstimator2DSeparationMethod(data=self._data,
                                                  length_options=self._signal_length_options,
                                                  signal_area_fraction_boundaries=(0.1, 0.4),
                                                  signal_num_of_occurrences_boundaries=(10, 150),
                                                  num_of_power_options=7,
                                                  signal_filter_gen=signal_2d_filter_gen,
                                                  noise_mean=self._noise_mean,
                                                  noise_std=self._noise_std,
                                                  signal_power_estimator_method=signal_power_estimator_method,
                                                  exp_attr=exp_attr,
                                                  logs=self._logs)
        else:
            print(f'Estimating signal length using curves method')
            self._length_estimator = LengthEstimator2DCurvesMethod(data=self._data,
                                                                   length_options=self._signal_length_options,
                                                                   signal_filter_gen=signal_1d_filter_gen,
                                                                   noise_mean=self._noise_mean,
                                                                   noise_std=self._noise_std,
                                                                   signal_power_estimator_method=signal_power_estimator_method,
                                                                   exp_attr=exp_attr,
                                                                   logs=self._logs)

    def run(self):
        start_time = time.time()
        likelihoods, most_likely_length = self._length_estimator.estimate()
        end_time = time.time()

        self._results = {
            "likelihoods": likelihoods,
            "most_likely_length": most_likely_length,
            "total_time": end_time - start_time
        }

        self.save_and_plot()

        return likelihoods

    def save_and_plot(self):
        plt.title(
            f"N={self._rows}, M={self._columns}, D={self._signal_length}, K={self._num_of_occurrences}, Noise Mean={self._noise_mean}, Noise STD={self._noise_std} \n"
            f"Signal Power Estimator Method={self._signal_power_estimator_method.name},\n"
            f"Most likely D={self._results['most_likely_length']}, Took {'%.3f' % (self._results['total_time'])} Seconds")

        likelihoods = self._results['likelihoods']
        plt.figure(1)
        for i, key in enumerate(likelihoods):
            plt.plot(self._signal_length_options, likelihoods[key], label=key)
            plt.legend(loc="upper right")
        plt.tight_layout()

        if self._save:
            fig_path = os.path.join(ROOT_DIR, f'src/experiments/plots/{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()


def __main__():
    sim_data = DataSimulator2D(rows=2000,
                               columns=2000,
                               signal_length=200,
                               signal_power=1,
                               signal_fraction=1 / 5,
                               signal_gen=lambda d, p: Shapes2D.disk(d, p),
                               noise_std=5,
                               noise_mean=0)

    Experiment2D(
        # mrc=MICROGRAPHS['simple_3'],
        simulator=sim_data,
        estimation_method=EstimationMethod.WellSeparation,
        name="std-10",
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        length_options=np.arange(50, 251, 20),
        plot=True,
        save=False
    ).run()


if __name__ == '__main__':
    __main__()
