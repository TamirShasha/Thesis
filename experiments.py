import numpy as np
import matplotlib.pyplot as plt
import time

from utils import create_random_signal_mask, add_pulses, add_gaus_noise
from algorithm import LengthExtractor

np.random.seed(500)


class Experiment:

    def __init__(self,
                 n=1000,
                 d=12,
                 k=20,
                 signal_avg_power=1,
                 noise_mean=0,
                 noise_std=0.3,
                 logs=False):
        self._n = n  # Data length
        self._d = d  # Signal Length
        self._k = k  # Num of signal occurrences
        self._signal_avg_power = signal_avg_power  # Signal Expected Avg Power
        self._noise_mean = noise_mean  # Expected noise mean
        self._noise_std = noise_std  # Expected noise std
        self._logs = logs

        signal_mask = create_random_signal_mask(self._k + 1, self._n - self._d * self._k)
        self._signal = np.full(self._d, self._signal_avg_power)
        self._clean_y = np.zeros(self._n)
        self._y_with_signals = add_pulses(self._clean_y, signal_mask, self._signal)
        self._noisy_y = add_gaus_noise(self._y_with_signals, self._noise_mean, self._noise_std)
        self._y = self._noisy_y

        self._signal_length_options = np.arange(self._d // 2, self._d * 2)

        self._length_extractor = LengthExtractor(y=self._y,
                                                 length_options=self._signal_length_options,
                                                 signal_avg_power=signal_avg_power,
                                                 noise_mean=self._noise_mean,
                                                 noise_std=self._noise_std,
                                                 logs=self._logs
                                                 )

    def _plot_likelihoods(self, likelihoods):
        plt.plot(self._signal_length_options, likelihoods)
        plt.show()

    def run(self):
        start_time = time.time()
        likelihoods, d = self._length_extractor.fast_extract()
        end_time = time.time()
        print(f'Most Likely D = {d} [D={self._d}], took: {end_time - start_time} seconds,', )

        self._plot_likelihoods(likelihoods)


def __main__():
    experiment = Experiment(
        n=300,
        d=10,
        k=5,
        signal_avg_power=1,
        noise_mean=0,
        noise_std=0.3,
        logs=False
    )
    experiment.run()


if __name__ == '__main__':
    __main__()
