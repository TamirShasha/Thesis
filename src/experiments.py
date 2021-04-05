import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR

from src.utils import create_random_k_tuple_sum_to_n, add_pulses, add_gaus_noise
from src.algorithm import LengthExtractor

np.random.seed(500)


class Experiment:

    def __init__(self,
                 name=str(time.time()),
                 n=1000,
                 d=12,
                 k=20,
                 noise_mean=0,
                 noise_std=0.3,
                 signal_fn=lambda d: np.full(d, 1),
                 signal_filter_gen=None,
                 use_exact_signal_power=False,
                 length_options=None,
                 save=True,
                 logs=True):
        self._name = name
        self._signal_fn = signal_fn
        self._n = n  # Data length
        self._d = d  # Signal Length
        self._k = k  # Num of signal occurrences
        self._signal_filter_gen = signal_filter_gen  # Signal generator per length d
        self._use_exact_signal_power = use_exact_signal_power
        self._noise_mean = noise_mean  # Expected noise mean
        self._noise_std = noise_std  # Expected noise std
        self._save = save
        self._logs = logs
        self._results = {}

        self._signal = self._signal_fn(self._d)
        if signal_filter_gen is None:
            self._signal_filter_gen = self._signal_fn
        signal_mask = create_random_k_tuple_sum_to_n(self._n - self._d * self._k, self._k + 1)
        self._clean_y = np.zeros(self._n)
        self._y_with_signals = add_pulses(self._clean_y, signal_mask, self._signal)
        self._y = add_gaus_noise(self._y_with_signals, self._noise_mean, self._noise_std)

        if length_options is None:
            length_options = np.arange(self._d // 4, int(self._d * 2), 10)
        self._signal_length_options = length_options

        exp_attr = {
            "d": self._d,
            "k": self._k,
            "use_exact_signal_power": self._use_exact_signal_power
        }
        self._length_extractor = LengthExtractor(y=self._y,
                                                 length_options=self._signal_length_options,
                                                 signal_filter_gen=self._signal_filter_gen,
                                                 noise_mean=self._noise_mean,
                                                 noise_std=self._noise_std,
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

        return likelihoods

    def plot_results(self):
        plt.title(f"N={self._n}, D={self._d}, K={self._k}, Noise Mean={self._noise_mean}, Noise STD={self._noise_std} \n"
                  f"Signal Exact Power={self._use_exact_signal_power},\n"
                  f"Most likely D={self._results['d']}, Took {'%.3f' % (self._results['total_time'])} Seconds")
        plt.plot(self._signal_length_options, self._results['likelihoods'])
        plt.tight_layout()

        if self._save:
            fig_path = os.path.join(ROOT_DIR, f'experiments_results/{self._name}.png')
            plt.savefig(fname=fig_path)

        plt.show()


def __main__():
    experiment = Experiment(
        name="using approx (default) filter",
        signal_fn=lambda d: np.sin(np.arange(d)) * 0.2 + 1,
        n=50000,
        d=50,
        k=300,
        noise_std=2.5,
        use_exact_signal_power=False,
        save=False
    )
    experiment.run()
    experiment.plot_results()


if __name__ == '__main__':
    __main__()
