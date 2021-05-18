import numpy as np
import matplotlib.pyplot as plt
import time
import os

from src.constants import ROOT_DIR
from src.utils import mrc
from src.algorithms.length_estimator_1d import SignalPowerEstimator
from src.algorithms.length_estimator_2d_curves_method import LengthExtractor2DCurvesMethod
from src.algorithms.very_well_separated_2d import LengthExtractor2D as VWS_LengthExtractor2D
from src.experiments.data_simulator_2d import Shapes2D

np.random.seed(500)


class RealDataExperiment:

    def __init__(self,
                 mrc_path,
                 name=str(time.time()),
                 real_length=12,
                 noise_mean=0,
                 noise_std=0.3,
                 signal_1d_filter_gen=lambda d: np.full(d, 1),
                 signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
                 length_options=None,
                 plot=True,
                 save=True,
                 logs=True):
        self._mrc_path = mrc_path
        self._data = mrc.read_mrc(mrc_path)
        self._name = mrc_path
        self._n = self._data.shape[0]  # Rows
        self._m = self._data.shape[1]  # Columns
        self._d = real_length  # Signal Length
        self._signal_1d_filter_gen = signal_1d_filter_gen  # Signal generator per length d
        self._signal_power_estimator_method = signal_power_estimator_method  # The method the algorithm uses to estimate signal power
        self._noise_mean = noise_mean  # Expected noise mean
        self._noise_std = noise_std  # Expected noise std

        self._plot = plot
        self._save = save
        self._logs = logs
        self._results = {}

        if length_options is None:
            length_options = np.arange(self._d // 4, int(self._d * 1.3), 3)
        self._signal_length_options = length_options

        exp_attr = {
            "d": self._d,
        }

        # plt.imshow(self._data, cmap='gray')
        # plt.show()

        self._length_extractor_tamir = LengthExtractor2DCurvesMethod(data=self._data,
                                                                     length_options=self._signal_length_options,
                                                                     signal_filter_gen=self._signal_1d_filter_gen,
                                                                     noise_mean=self._noise_mean,
                                                                     noise_std=self._noise_std,
                                                                     signal_power_estimator_method=self._signal_power_estimator_method,
                                                                     exp_attr=exp_attr,
                                                                     logs=self._logs)

        self._length_extractor = VWS_LengthExtractor2D(y=self._data,
                                                       length_options=self._signal_length_options,
                                                       signal_filter_gen=self._signal_1d_filter_gen,
                                                       noise_mean=self._noise_mean,
                                                       noise_std=self._noise_std,
                                                       signal_power_estimator_method=self._signal_power_estimator_method,
                                                       exp_attr=exp_attr,
                                                       logs=self._logs)

    def run(self):
        start_time = time.time()
        likelihoods, best_ds = self._length_extractor_tamir.estimate()
        # likelihoods = {}
        # best_ds = {}
        # vws_likelihoods, vws_best_ds = self._length_extractor.extract()
        # likelihoods['vws'] = vws_likelihoods
        # best_ds['vws'] = vws_best_ds
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
            f"N={self._n}, M={self._m}, D={self._d}, Noise Mean={self._noise_mean}, Noise STD={self._noise_std} \n"
            f"Signal Power Estimator Method={self._signal_power_estimator_method.name},\n"
            f"Most likely D=NONE, Took {'%.3f' % (self._results['total_time'])} Seconds")

        likelihoods = self._results['likelihoods']
        plt.figure(1)
        for i, key in enumerate(likelihoods):
            plt.plot(self._signal_length_options, likelihoods[key], label=key)
            plt.legend(loc="upper right")

        # best_ds = self._results['best_ds']
        # plt.figure(2)
        # for i, key in enumerate(best_ds):
        #     error = np.abs(np.array(best_ds[key]) - self._d)
        #     plt.plot(np.arange(len(error)), error, label=key)
        #     plt.legend(loc="upper right")
        #
        # plt.tight_layout()

        if self._save:
            fig_path = os.path.join(ROOT_DIR, f'src/experiments/plots/{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()


def __main__():
    d = 301
    RealDataExperiment(
        mrc_path=os.path.join(ROOT_DIR, os.pardir, 'data', 'clean_one_std_3.mrc'),
        name="blalba",
        real_length=d,
        # signal_1d_filter_gen=lambda l: Shapes2D.disk(l, 1),
        signal_1d_filter_gen=lambda l: np.full(l, 1),
        # length_options=[300, 400],
        length_options=np.arange(100, 350, 20),
        noise_std=3,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        plot=True,
        save=False
    ).run()


if __name__ == '__main__':
    __main__()
