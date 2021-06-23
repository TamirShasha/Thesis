import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR
import logging
from datetime import datetime
import pathlib

from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.algorithms.length_estimator_1d import SignalPowerEstimator
from src.algorithms.length_estimator_2d import LengthEstimator2D, EstimationMethod
from src.experiments.micrograph import Micrograph, MICROGRAPHS
from src.experiments.particles_projections import PARTICLE_250
from src.utils.logger import logger

# np.random.seed(500)
logger.setLevel(logging.INFO)


class Experiment2D:

    def __init__(self,
                 name=str(time.time()),
                 mrc: Micrograph = None,
                 simulator=None,
                 signal_2d_filter_gen=lambda d, p=1: Shapes2D.disk(d, p),
                 signal_1d_filter_gen=lambda d, p=1: np.full(d, p),
                 signal_power_estimator_method=SignalPowerEstimator.Exact,
                 length_options=None,
                 signal_area_coverage_boundaries=(0.05, 0.4),
                 signal_num_of_occurrences_boundaries=(10, 150),
                 num_of_power_options=10,
                 estimation_method: EstimationMethod = EstimationMethod.WellSeparation,
                 plot=True,
                 save=True,
                 save_dir=os.path.join(ROOT_DIR, f'src/experiments/plots/'),
                 logs=True):
        self._name = name  # Experiment name
        self._signal_power_estimator_method = signal_power_estimator_method
        self._estimation_method = estimation_method
        self._data_simulator = simulator
        self._mrc = mrc

        self._plot = plot
        self._save = save
        self._save_dir = save_dir
        self._logs = logs
        self._results = {}

        if mrc is None:
            logger.info(f'Simulating data, number of occurrences is {simulator.occurrences}')
            if simulator is None:
                simulator = DataSimulator2D()

            self._data = simulator.simulate()
            self._num_of_occurrences = simulator.occurrences
            self._noise_std = simulator.noise_std
            self._noise_mean = simulator.noise_mean
            self._signal_length = simulator.signal_length
            self._applied_ctf = simulator.apply_ctf
        else:
            logger.info(f'Loading given micrograph from {mrc.name}')
            self._data = mrc.load_mrc()
            self._num_of_occurrences = mrc.occurrences
            self._noise_std = mrc.noise_std
            self._noise_mean = mrc.noise_mean
            self._signal_length = mrc.signal_length
            self._applied_ctf = True

        self._rows = self._data.shape[0]
        self._columns = self._data.shape[1]

        plt.rcParams["figure.figsize"] = (16, 9)
        if self._plot:
            plt.imshow(self._data, cmap='gray')
            plt.show()

        if length_options is None:
            length_options = np.arange(self._signal_length // 4, int(self._signal_length), 10)
        self._signal_length_options = length_options

        self._exp_attr = {
            "d": self._signal_length,
            "k": self._num_of_occurrences,
        }

        self._length_estimator = LengthEstimator2D(data=self._data,
                                                   length_options=self._signal_length_options,
                                                   signal_area_fraction_boundaries=signal_area_coverage_boundaries,
                                                   signal_num_of_occurrences_boundaries=signal_num_of_occurrences_boundaries,
                                                   num_of_power_options=num_of_power_options,
                                                   signal_filter_gen_1d=signal_1d_filter_gen,
                                                   signal_filter_gen_2d=signal_2d_filter_gen,
                                                   noise_mean=self._noise_mean,
                                                   noise_std=self._noise_std,
                                                   signal_power_estimator_method=signal_power_estimator_method,
                                                   estimation_method=estimation_method,
                                                   exp_attr=self._exp_attr,
                                                   logs=self._logs,
                                                   plot=self._plot)

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
        fig, ((mrc, results), (particle, dpk_table)) = plt.subplots(2, 2)

        title = f"MRC size=({self._rows}, {self._columns}), " \
                f"Signal length={self._signal_length}\n"

        if self._mrc is None:
            title += f"Signal power={self._data_simulator.signal_power}, " \
                     f"Signal area coverage={int(np.round(self._data_simulator.signal_fraction, 2) * 100)}% \n" \
                     f"SNR={self._data_simulator.snr}db "
        else:
            title += f"MRC={self._mrc.name}\n"

        title += f"Noise\u007E\u2115({self._noise_mean}, {self._noise_std}), " \
                 f"CTF={self._applied_ctf}\n" \
                 f"SPE={self._signal_power_estimator_method}, " \
                 f"Estimation method={self._estimation_method.name}\n" \
                 f"Most likely length={self._results['most_likely_length']}, " \
                 f"Took {int(self._results['total_time'])} seconds"
        fig.suptitle(title)

        mrc.imshow(self._data, cmap='gray')

        likelihoods = self._results['likelihoods']
        for i, key in enumerate(likelihoods):
            if key == 'max':
                pass
            results.plot(self._signal_length_options, likelihoods[key], label=key, alpha=0.3)

        results.set_xlabel('Lengths')
        results.set_ylabel('Likelihood')

        results_max = results.twinx()
        results_max.plot(self._signal_length_options, likelihoods['max'], label='max')

        results.legend(loc="lower right")

        if self._data_simulator:
            particle.imshow(self._data_simulator.create_signal_instance(), cmap='gray')

        self._length_estimator.generate_dpk_plot_table(dpk_table)

        fig.tight_layout()

        if self._save:
            curr_date = str(datetime.now().strftime("%d-%m-%Y"))
            dir_path = os.path.join(self._save_dir, curr_date)
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

            curr_time = str(datetime.now().strftime("%H-%M-%S"))
            fig_path = os.path.join(dir_path, f'{curr_time}_{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()


def __main__():
    sim_data = DataSimulator2D(rows=4000,
                               columns=4000,
                               signal_length=250,
                               signal_power=1,
                               signal_fraction=1 / 6,
                               # signal_gen=PARTICLE_250.get_signal_gen(),
                               signal_gen=Shapes2D.sphere,
                               noise_std=1,
                               noise_mean=0,
                               apply_ctf=False)

    Experiment2D(
        # mrc=MICROGRAPHS['whitened002_x10'],
        name=f"expy",
        simulator=sim_data,
        estimation_method=EstimationMethod.Curves,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        length_options=np.arange(50, 350, 10),
        signal_num_of_occurrences_boundaries=(20, 150),
        signal_area_coverage_boundaries=(0.05, 0.20),
        num_of_power_options=10,
        plot=True,
        save=True
    ).run()


if __name__ == '__main__':
    __main__()
