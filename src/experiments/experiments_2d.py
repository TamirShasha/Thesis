import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR
from enum import Enum

from src.experiments.data_simulator_2d import simulate_data, Shapes2D
from src.algorithms.length_estimator_1d import SignalPowerEstimator
from src.algorithms.length_estimator_2d_sep_method import LengthEstimator2DSeparationMethod
from src.algorithms.length_estimator_2d_curves_method import LengthEstimator2DCurvesMethod
from src.utils.mrc import read_mrc

np.random.seed(405)


class EstimationMethod(Enum):
    Curves = 0,
    WellSeparation = 1


class Experiment2D:

    def __init__(self,
                 mrc_name=None,
                 name=str(time.time()),
                 rows=1000,
                 columns=1000,
                 signal_length=12,
                 num_of_occurrences=20,
                 signal_fraction=None,
                 noise_mean=0,
                 noise_std=0.3,
                 signal_gen=None,
                 signal_shape=None,
                 signal_2d_filter_gen=lambda d: Shapes2D.disk(d, 1),
                 signal_1d_filter_gen=lambda d: np.full(d, 1),
                 signal_power_estimator_method=SignalPowerEstimator.Exact,
                 length_options=None,
                 method: EstimationMethod = EstimationMethod.WellSeparation,
                 plot=True,
                 save=True,
                 logs=True):
        self._mrc_path = os.path.join(ROOT_DIR, os.pardir, 'data', mrc_name)  # Path to micrograph
        self._name = name  # Experiment name
        self._rows = rows  # Rows
        self._columns = columns  # Columns
        self._signal_length = signal_length  # Signal Length
        self._num_of_occurrences = num_of_occurrences  # Num of signal occurrences
        self._signal_gen = signal_gen
        self._signal_shape = signal_shape
        self._signal_fraction = signal_fraction  # The fraction of signal out of all data
        self._signal_1d_filter_gen = signal_1d_filter_gen  # Signal generator per length d for 1d
        self._signal_2d_filter_gen = signal_2d_filter_gen  # Signal generator per length d for 2d
        self._signal_power_estimator_method = signal_power_estimator_method  # The method the algorithm uses to estimate signal power
        self._noise_mean = noise_mean  # Expected noise mean
        self._noise_std = noise_std  # Expected noise std
        self._estimation_method = method  # Estimation method

        self._plot = plot
        self._save = save
        self._logs = logs
        self._results = {}

        if self._signal_shape is None:
            self._signal_shape = (self._signal_length, self._signal_length)

        if self._signal_fraction:
            self._num_of_occurrences = int(
                (self._rows * self._columns / np.prod(self._signal_shape)) * self._signal_fraction)

        if self._signal_gen is None:
            self._signal_gen = lambda: Shapes2D.square(self._signal_length, 1)

        if mrc_name is None:
            print(f'Simulating data, number of occurrences is {self._num_of_occurrences}')
            self._data = simulate_data((rows, columns), self._signal_gen, self._signal_shape, self._num_of_occurrences,
                                       self._noise_std, self._noise_mean)
        else:
            print(f'Loading given micrograph from {mrc_name}')
            self._data = read_mrc(self._mrc_path)

        if length_options is None:
            length_options = np.arange(self._signal_length // 4, int(self._signal_length), 10)
        self._signal_length_options = length_options

        exp_attr = {
            "d": self._signal_length,
            "k": self._num_of_occurrences,
        }

        plt.imshow(self._data, cmap='gray')
        plt.show()

        if self._estimation_method == EstimationMethod.WellSeparation:
            print(f'Estimating signal length using well separation method')
            self._length_estimator = LengthEstimator2DSeparationMethod(data=self._data,
                                                                       length_options=self._signal_length_options,
                                                                       signal_filter_gen=self._signal_2d_filter_gen,
                                                                       noise_mean=self._noise_mean,
                                                                       noise_std=self._noise_std,
                                                                       signal_power_estimator_method=self._signal_power_estimator_method,
                                                                       exp_attr=exp_attr,
                                                                       logs=self._logs)
        else:
            print(f'Estimating signal length using curves method')
            self._length_estimator = LengthEstimator2DCurvesMethod(data=self._data,
                                                                   length_options=self._signal_length_options,
                                                                   signal_filter_gen=self._signal_1d_filter_gen,
                                                                   noise_mean=self._noise_mean,
                                                                   noise_std=self._noise_std,
                                                                   signal_power_estimator_method=self._signal_power_estimator_method,
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
    Experiment2D(
        mrc_name='clean_one_std_3.mrc',
        method=EstimationMethod.WellSeparation,
        name="std-10",
        rows=1000,
        columns=1000,
        signal_length=100,
        signal_fraction=1 / 4,
        signal_gen=lambda: Shapes2D.disk(100, 1),
        noise_std=3,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        length_options=np.arange(90, 101, 10),
        plot=True,
        save=False
    ).run()


if __name__ == '__main__':
    __main__()
