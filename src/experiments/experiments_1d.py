import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR

from src.algorithms.utils import create_random_k_tuple_sum_to_n
from src.experiments.data_simulator_1d import add_pulses, add_gaus_noise
from src.algorithms.length_extractor_1d import LengthExtractor1D, SignalPowerEstimator


class Experiment:

    def __init__(self,
                 name=str(time.time()),
                 n=1000,
                 d=12,
                 k=20,
                 signal_fraction=None,
                 noise_mean=0,
                 noise_std=0.3,
                 signal_fn=lambda d: np.full(d, 1),
                 signal_filter_gen=None,
                 signal_power_estimator_method=SignalPowerEstimator.Exact,
                 length_options=None,
                 plot=True,
                 save=True,
                 logs=True):
        self._name = name
        self._signal_fn = signal_fn
        self._n = n  # Data length
        self._d = d  # Signal Length
        self._k = k  # Num of signal occurrences
        self._signal_fraction = signal_fraction  # The fraction of signal out of all data
        self._signal_filter_gen = signal_filter_gen  # Signal generator per length d
        self._signal_power_estimator_method = signal_power_estimator_method  # The method the algorithm uses to estimate signal power
        self._noise_mean = noise_mean  # Expected noise mean
        self._noise_std = noise_std  # Expected noise std

        self._plot = plot
        self._save = save
        self._logs = logs
        self._results = {}

        if self._signal_fraction:
            self._k = int((self._n / self._d) * self._signal_fraction)

        self._signal = self._signal_fn(self._d)
        if signal_filter_gen is None:
            self._signal_filter_gen = self._signal_fn

        print('Arranging data...')
        signal_mask = create_random_k_tuple_sum_to_n(self._n - self._d * self._k, self._k + 1)
        self._clean_y = np.zeros(self._n)
        self._y_with_signals = add_pulses(self._clean_y, signal_mask, self._signal)
        self._y = add_gaus_noise(self._y_with_signals, self._noise_mean, self._noise_std)

        if length_options is None:
            length_options = np.arange(self._d // 4, int(self._d * 2), 5)
        self._signal_length_options = length_options

        exp_attr = {
            "d": self._d,
            "k": self._k,
        }
        self._length_extractor = LengthExtractor1D(y=self._y,
                                                   length_options=self._signal_length_options,
                                                   signal_filter_gen=self._signal_filter_gen,
                                                   noise_mean=self._noise_mean,
                                                   noise_std=self._noise_std,
                                                   signal_power_estimator_method=self._signal_power_estimator_method,
                                                   exp_attr=exp_attr,
                                                   logs=self._logs)

    def run(self):
        start_time = time.time()
        likelihoods, d = self._length_extractor.extract()
        end_time = time.time()

        self._results = {
            "likelihoods": likelihoods,
            "d": d,
            "total_time": end_time - start_time
        }

        self.save_and_plot()

        return likelihoods

    def save_and_plot(self):
        plt.title(
            f"N={self._n}, D={self._d}, K={self._k}, Noise Mean={self._noise_mean}, Noise STD={self._noise_std} \n"
            f"Signal Power Estimator Method={self._signal_power_estimator_method.name},\n"
            f"Most likely D={self._results['d']}, Took {'%.3f' % (self._results['total_time'])} Seconds")
        plt.plot(self._signal_length_options, self._results['likelihoods'])
        plt.tight_layout()

        if self._save:
            fig_path = os.path.join(ROOT_DIR, f'src/experiments/plots/{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()


def __main__():
    # Experiment(
    #     name="std-5",
    #     n=300000,
    #     d=150,
    #     signal_fraction=1 / 4,
    #     noise_std=5,
    #     signal_power_estimator_method=SignalPowerEstimator.SecondMoment,
    #     plot=True,
    #     save=False
    # ).run()

    Experiment(
        name="std-10",
        n=70000,
        d=150,
        signal_fraction=1 / 5,
        noise_std=5,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        plot=True,
        save=False
    ).run()

    # Experiment(
    #     name="std-13",
    #     n=50000,
    #     d=100,
    #     signal_fraction=1 / 5,
    #     noise_std=7,
    #     signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
    #     plot=True,
    #     save=False
    # ).run()


if __name__ == '__main__':
    __main__()