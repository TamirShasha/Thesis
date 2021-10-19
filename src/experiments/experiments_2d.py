import numpy as np
import matplotlib.pyplot as plt
import time
import os
from src.constants import ROOT_DIR
import logging
from datetime import datetime
import pathlib
from enum import Enum

from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.algorithms.length_estimator_2d_curves_method import LengthEstimator2DCurvesMethod
from src.algorithms.length_estimator_2d_very_well_separated import LengthEstimator2DVeryWellSeparated
from src.experiments.micrograph import Micrograph
from src.experiments.particles_projections import PARTICLE_200
from src.utils.logger import logger

# np.random.seed(501)
logger.setLevel(logging.INFO)


class EstimationMethod(Enum):
    Curves = 0,
    VeryWellSeparated = 1


class Experiment2D:

    def __init__(self,
                 name=str(time.time()),
                 mrc: Micrograph = None,
                 simulator=None,
                 signal_1d_filter_gen=lambda d, p=1: np.full(d, p),
                 length_options=None,
                 filter_basis_size=20,
                 down_sample_size=1000,
                 fixed_num_of_occurrences=30,
                 estimation_method: EstimationMethod = EstimationMethod.VeryWellSeparated,
                 plot=True,
                 save=True,
                 save_dir=os.path.join(ROOT_DIR, f'src/experiments/plots/'),
                 logs=True):
        self._name = name  # Experiment name
        self._estimation_method = estimation_method
        self._data_simulator = simulator
        self._mrc = mrc
        self._filter_basis_size = filter_basis_size
        self._fixed_num_of_occurrences = fixed_num_of_occurrences
        self._down_sample_size = down_sample_size

        self._plot = plot
        self._save = save
        self._save_dir = save_dir
        self._logs = logs
        self._results = {}

        curr_date = str(datetime.now().strftime("%d-%m-%Y"))
        curr_time = str(datetime.now().strftime("%H-%M-%S"))
        self.experiment_dir = os.path.join(self._save_dir, curr_date, curr_time)
        pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        if mrc is None:
            logger.info(f'Simulating data, number of occurrences is {simulator.occurrences}')
            if simulator is None:
                simulator = DataSimulator2D()

            self._data = simulator.simulate()
            self._noise_std = simulator.noise_std
            self._noise_mean = simulator.noise_mean
            self._signal_length = simulator.signal_length
            self._applied_ctf = simulator.apply_ctf
        else:
            logger.info(f'Loading given micrograph from {mrc.name}')
            self._data = mrc.get_micrograph()
            # self._data = self._data[:min(self._data.shape), :min(self._data.shape)]
            self._noise_std = mrc.noise_std
            self._noise_mean = mrc.noise_mean
            self._signal_length = None
            self._applied_ctf = True

        self._rows = self._data.shape[0]
        self._columns = self._data.shape[1]

        plt.rcParams["figure.figsize"] = (16, 9)
        if self._plot:
            plt.imshow(self._data, cmap='gray')
            plt.show()

        if length_options is None:
            length_options = np.arange(100, 801, 100)
        self._signal_length_options = length_options

        if self._estimation_method == EstimationMethod.Curves:
            logger.info(f'Estimating signal length using Curves method')
            self._length_estimator = LengthEstimator2DCurvesMethod(data=self._data,
                                                                   length_options=self._signal_length_options,
                                                                   signal_filter_gen_1d=signal_1d_filter_gen,
                                                                   noise_mean=self._noise_mean,
                                                                   noise_std=self._noise_std,
                                                                   logs=self._logs,
                                                                   experiment_dir=self.experiment_dir)
        else:
            self._length_estimator = LengthEstimator2DVeryWellSeparated(data=self._data,
                                                                        length_options=self._signal_length_options,
                                                                        fixed_num_of_occurrences=self._fixed_num_of_occurrences,
                                                                        noise_mean=self._noise_mean,
                                                                        noise_std=self._noise_std,
                                                                        downsample_to_num_of_rows=self._down_sample_size,
                                                                        filter_basis_size=self._filter_basis_size,
                                                                        logs=self._logs,
                                                                        plots=self._plot,
                                                                        save=self._save,
                                                                        experiment_dir=self.experiment_dir)

    def run(self):
        start_time = time.time()
        likelihoods, optimal_coeffs, estimated_signal = self._length_estimator.estimate()
        end_time = time.time()

        self._results = {
            "likelihoods": likelihoods,
            "optimal_coeffs": optimal_coeffs,
            "estimated_signal": estimated_signal,
            "total_time": end_time - start_time
        }

        self.save_and_plot()

        return likelihoods

    def save_and_plot(self):

        likelihoods = self._results['likelihoods']
        most_likely_index = np.nanargmax(likelihoods)

        fig = plt.figure()

        mrc_fig = plt.subplot2grid((2, 3), (0, 2))
        likelihoods_fig = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        particle_fig = plt.subplot2grid((2, 3), (1, 0))
        est_particle_fig = plt.subplot2grid((2, 3), (1, 1))
        est_basis_coeffs_fig = plt.subplot2grid((2, 3), (1, 2))

        title = f"MRC size=({self._rows}, {self._columns}), " \
                f"Signal length={self._signal_length}, " \
                f"Noise\u007E\u2115({self._noise_mean}, {self._noise_std})\n"

        if self._mrc is None:
            title += f"Signal power={self._data_simulator.signal_power}, " \
                     f"Signal area coverage={int(np.round(self._data_simulator.signal_fraction, 2) * 100)}% \n" \
                     f"SNR={self._data_simulator.snr}db (MRC-SNR={self._data_simulator.mrc_snr}db), "
        else:
            title += f"MRC={self._mrc.name}\n"

        title += f"CTF={self._applied_ctf}, " \
                 f"Estimation method={self._estimation_method.name}\n" \
                 f"Most likely length={self._signal_length_options[most_likely_index]}, " \
                 f"Took {int(self._results['total_time'])} seconds"
        fig.suptitle(title)

        mrc_fig.imshow(self._data, cmap='gray')
        est_particle_fig.imshow(self._results['estimated_signal'], cmap='gray')

        most_likely_coeffs = self._results['optimal_coeffs'][most_likely_index]
        est_basis_coeffs_fig.bar(np.arange(len(most_likely_coeffs)), most_likely_coeffs)

        likelihoods_fig.plot(self._signal_length_options, likelihoods)
        likelihoods_fig.set_xlabel('Lengths')
        likelihoods_fig.set_ylabel('Likelihood')

        if self._data_simulator:
            particle_fig.imshow(self._data_simulator.create_signal_instance(), cmap='gray')

        if self._save:
            # curr_time = str(datetime.now().strftime("%H-%M-%S"))
            fig_path = os.path.join(self.experiment_dir, f'{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()


def __main__():
    sim_data = DataSimulator2D(rows=1000,
                               columns=1000,
                               signal_length=80,
                               signal_power=1,
                               signal_fraction=1 / 6,
                               # signal_gen=Shapes2D.sphere,
                               # signal_gen=lambda l, p: Shapes2D.double_disk(l, l // 2, p, 0),
                               signal_gen=Shapes2D.sphere,
                               noise_std=10,
                               noise_mean=0,
                               apply_ctf=False)

    Experiment2D(
        name=f"expy",
        # mrc=MICROGRAPHS['002_whitened'],
        # mrc=Micrograph('Tamir', 300, 'C:\\Users\\tamir\\Desktop\\תזה\\data\\001_raw.mat'),
        simulator=sim_data,
        estimation_method=EstimationMethod.VeryWellSeparated,
        length_options=np.array([40, 60, 80, 100, 120]),
        plot=True,
        save=True
    ).run()


if __name__ == '__main__':
    __main__()
