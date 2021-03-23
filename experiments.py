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
                 signal_seperation=0,
                 noise_mean=0,
                 noise_std=0.3,
                 use_exact_signal_power=False,
                 use_exact_k=False,
                 save=False,
                 logs=False):
        self._n = n  # Data length
        self._d = d  # Signal Length
        self._k = k  # Num of signal occurrences
        self._signal_avg_power = signal_avg_power  # Signal Expected Avg Power
        self._signal_seperation = signal_seperation
        self._use_exact_signal_power = use_exact_signal_power
        self._use_exact_k = use_exact_k
        self._noise_mean = noise_mean  # Expected noise mean
        self._noise_std = noise_std  # Expected noise std
        self._save = save
        self._logs = logs
        self._results = {}

        signal_mask = create_random_signal_mask(self._k + 1, self._n - self._d * self._k)
        self._signal = np.full(self._d, self._signal_avg_power)
        self._clean_y = np.zeros(self._n)
        self._y_with_signals = add_pulses(self._clean_y, signal_mask, self._signal)
        self._noisy_y = add_gaus_noise(self._y_with_signals, self._noise_mean, self._noise_std)
        self._y = self._noisy_y

        self._signal_length_options = np.arange(self._d // 4, int(self._d * 2), 4)

        exp_attr = {
            "d": self._d,
            "k": self._k,
            "use_exact_signal_power": self._use_exact_signal_power,
            "use_exact_k": self._use_exact_k
        }
        self._length_extractor = LengthExtractor(y=self._y,
                                                 length_options=self._signal_length_options,
                                                 signal_avg_power=self._signal_avg_power,
                                                 signal_seperation=self._signal_seperation,
                                                 noise_mean=self._noise_mean,
                                                 noise_std=self._noise_std,
                                                 exp_attr=exp_attr,
                                                 logs=self._logs
                                                 )

    def run(self):
        start_time = time.time()
        likelihoods, d = self._length_extractor.extract()
        end_time = time.time()

        self._results = {
            "likelihoods": likelihoods,
            "d": d,
            "total_time": end_time - start_time
        }

    def plot_results(self):
        plt.title(f"N={self._n}, D={self._d}, K={self._k}, NoiseMean={self._noise_mean}, NoiseSTD={self._noise_std} "
                  f" Seperation={self._signal_seperation}, ExactPower={self._use_exact_signal_power}, ExactK={self._use_exact_k}\n"
                  f"\n Most likely D={self._results['d']}, Took {'%.3f' % (self._results['total_time'])} Seconds")
        plt.plot(self._signal_length_options, self._results['likelihoods'])

        if self._save:
            plt.savefig('experiments_results/' + str(time.time()) + '.pdf')

        plt.show()


def __main__():
    experiment1 = Experiment(
        n=2000,
        d=20,
        k=30,
        signal_avg_power=1,
        noise_mean=0,
        noise_std=0.5,
        logs=True
    )
    # experiment1.run()
    # experiment1.plot_results()

    experiment2 = Experiment(
        n=1000,
        d=20,
        k=20,
        signal_avg_power=1,
        noise_mean=0,
        noise_std=1,
        logs=True
    )
    # experiment2.run()
    # experiment2.plot_results()

    experiment3 = Experiment(
        n=3000,
        d=50,
        k=20,
        signal_avg_power=1,
        noise_mean=0,
        noise_std=1,
        logs=True
    )
    # experiment3.run()
    # experiment3.plot_results()

    experiment4 = Experiment(
        n=10000,
        d=50,
        k=70,
        signal_avg_power=1,
        signal_seperation=0,
        noise_mean=0,
        noise_std=1.5,
        use_exact_k=False,
        use_exact_signal_power=True,
        logs=True,
        save=True
    )
    experiment4.run()
    experiment4.plot_results()


if __name__ == '__main__':
    __main__()
