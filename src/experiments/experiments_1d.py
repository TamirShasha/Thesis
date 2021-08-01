import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR
import logging
from datetime import datetime

from src.algorithms.utils import generate_random_signal_positions
from src.experiments.data_simulator_1d import add_pulses, add_gaus_noise
from src.algorithms.length_estimator_1d import LengthEstimator1D, SignalPowerEstimator
from src.utils.logger import logger

np.random.seed(500)
logger.setLevel(logging.INFO)


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
                 save_dir=os.path.join(ROOT_DIR, f'src/experiments/plots/'),
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
        self._save_dir = save_dir
        self._logs = logs
        self._results = {}

        if self._signal_fraction:
            self._k = int((self._n / self._d) * self._signal_fraction)

        self._signal = self._signal_fn(self._d)
        if signal_filter_gen is None:
            self._signal_filter_gen = self._signal_fn

        logger.info('Arranging data...')
        signal_mask = generate_random_signal_positions(self._n - self._d * self._k, self._k + 1)
        self._clean_y = np.zeros(self._n)
        self._y_with_signals = add_pulses(self._clean_y, self._signal, signal_mask)
        self._y = add_gaus_noise(self._y_with_signals, self._noise_mean, self._noise_std)

        if length_options is None:
            length_options = np.arange(self._d // 4, int(self._d * 3), 10)
        self._signal_length_options = length_options

        exp_attr = {
            "d": self._d,
            "k": self._k,
        }
        self._length_extractor = LengthEstimator1D(data=self._y,
                                                   length_options=self._signal_length_options,
                                                   signal_filter_gen=self._signal_filter_gen,
                                                   noise_mean=self._noise_mean,
                                                   noise_std=self._noise_std,
                                                   signal_power_estimator_method=self._signal_power_estimator_method,
                                                   exp_attr=exp_attr,
                                                   logs=self._logs)

    def run(self):
        start_time = time.time()
        likelihoods, d = self._length_extractor.estimate()
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
            f"Signal Power Estimator Method={self._signal_power_estimator_method},\n"
            f"Most likely D={self._results['d']}, Took {'%.3f' % (self._results['total_time'])} Seconds")
        plt.plot(self._signal_length_options, self._results['likelihoods'])
        plt.tight_layout()

        if self._save:
            date_time = str(datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
            fig_path = os.path.join(self._save_dir, f'{date_time}_{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()


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
        n=40000,
        d=600,
        signal_fraction=1/5,
        signal_fn=lambda d: np.full(d, 1),
        signal_filter_gen=lambda d: np.full(d, 1),
        noise_std=8,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        plot=True,
        save=False
    ).run()

    # d = 100
    # len_ops = np.arange(d // 4, int(d * 1.5), 10)
    # # len_ops = [50]
    # likelihoods = []
    # names = []
    # # ps = np.arange(1, 3.1, 0.5)
    # ps = [1, 2]
    # for i, p in enumerate(ps):
    #     np.random.seed(500)
    #
    #     names.append(f'p_{p}')
    #     likelihoods.append(Experiment(
    #         name="std-10",
    #         n=3000,
    #         d=d,
    #         signal_fraction=1 / 5,
    #         length_options=len_ops,
    #         signal_fn=lambda d: np.full(d, 1),
    #         signal_filter_gen=lambda d: np.full(d, p),
    #         noise_std=5,
    #         signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
    #         plot=False,
    #         save=False
    #     ).run())
    #
    # plt.plot(len_ops, np.max(np.array(likelihoods), axis=0))
    # names.append('max')
    #
    # plt.legend(names)
    # plt.show()

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
