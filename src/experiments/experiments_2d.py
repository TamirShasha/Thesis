import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR

from src.experiments.data_simulator_2d import simulate_data, Shapes2D
from src.algorithms.length_estimator_1d import SignalPowerEstimator
from src.algorithms.length_estimator_2d import LengthExtractor2D

np.random.seed(500)


class Experiment2D:

    def __init__(self,
                 name=str(time.time()),
                 n=1000,
                 m=1000,
                 d=12,
                 k=20,
                 signal_fraction=None,
                 noise_mean=0,
                 noise_std=0.3,
                 signal_gen=None,
                 signal_shape=None,
                 signal_1d_filter_gen=lambda d: np.full(d, 1),
                 signal_power_estimator_method=SignalPowerEstimator.Exact,
                 length_options=None,
                 plot=True,
                 save=True,
                 logs=True):
        self._name = name
        self._n = n  # Rows
        self._m = m  # Columns
        self._d = d  # Signal Length
        self._k = k  # Num of signal occurrences
        self._signal_gen = signal_gen
        self._signal_shape = signal_shape
        self._signal_fraction = signal_fraction  # The fraction of signal out of all data
        self._signal_1d_filter_gen = signal_1d_filter_gen  # Signal generator per length d
        self._signal_power_estimator_method = signal_power_estimator_method  # The method the algorithm uses to estimate signal power
        self._noise_mean = noise_mean  # Expected noise mean
        self._noise_std = noise_std  # Expected noise std

        self._plot = plot
        self._save = save
        self._logs = logs
        self._results = {}

        if self._signal_shape is None:
            self._signal_shape = (self._d, self._d)

        if self._signal_fraction:
            self._k = int((self._n * self._m / np.prod(self._signal_shape)) * self._signal_fraction)

        if self._signal_gen is None:
            self._signal_gen = lambda: Shapes2D.square(self._d, 1)

        print(f'Arranging data: number of occurrences is {self._k}')
        self._y = simulate_data((n, m), self._signal_gen, self._signal_shape, self._k,
                                self._noise_std, self._noise_mean)

        if length_options is None:
            length_options = np.arange(self._d // 4, int(self._d * 1.3), 3)
        self._signal_length_options = length_options

        exp_attr = {
            "d": self._d,
            "k": self._k,
        }

        plt.imshow(self._y, cmap='gray')
        plt.show()

        self._length_extractor = LengthExtractor2D(y=self._y,
                                                   length_options=self._signal_length_options,
                                                   signal_filter_gen=self._signal_1d_filter_gen,
                                                   noise_mean=self._noise_mean,
                                                   noise_std=self._noise_std,
                                                   signal_power_estimator_method=self._signal_power_estimator_method,
                                                   exp_attr=exp_attr,
                                                   logs=self._logs)

    def run(self):
        start_time = time.time()
        likelihoods, best_ds = self._length_extractor.extract()
        end_time = time.time()

        self._results = {
            "likelihoods": likelihoods,
            "best_ds": best_ds,
            "total_time": end_time - start_time
        }

        self.save_and_plot()

        return None

    def save_and_plot(self):
        plt.title(
            f"N={self._n}, M={self._m}, D={self._d}, K={self._k}, Noise Mean={self._noise_mean}, Noise STD={self._noise_std} \n"
            f"Signal Power Estimator Method={self._signal_power_estimator_method.name},\n"
            f"Most likely D=NONE, Took {'%.3f' % (self._results['total_time'])} Seconds")

        # likelihoods = self._results['likelihoods']
        # i = 1
        # for key in likelihoods:
        #     plt.figure(i)
        #     plt.plot(self._signal_length_options, likelihoods[key], label=key)
        #     plt.legend(loc="upper right")
        #     i += 1

        likelihoods = self._results['likelihoods']
        plt.figure(1)
        for i, key in enumerate(likelihoods):
            plt.plot(self._signal_length_options, likelihoods[key], label=key)
            plt.legend(loc="upper right")

        best_ds = self._results['best_ds']
        plt.figure(2)
        for i, key in enumerate(best_ds):
            error = np.abs(np.array(best_ds[key]) - self._d)
            plt.plot(np.arange(len(error)), error, label=key)
            plt.legend(loc="upper right")

        plt.tight_layout()

        if self._save:
            fig_path = os.path.join(ROOT_DIR, f'src/experiments/plots/{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()


def __main__():
    d = 100
    Experiment2D(
        name="std-10",
        n=2000,
        m=2000,
        d=d,
        signal_fraction=1 / 6,
        # signal_gen=lambda: Shapes2D.ellipse(d, int(2 * d / 3), 1),
        signal_gen=lambda: Shapes2D.disk(d, 1),
        length_options=np.arange(int(d // 4), int(d * 2), 20),
        # length_options=[40, 50, 60],
        noise_std=5,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        plot=True,
        save=False
    ).run()


if __name__ == '__main__':
    __main__()
