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
from src.utils.micrograph import Micrograph, cryo_downsample
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
                 signal_length_by_percentage=None,
                 filter_basis_size=20,
                 down_sample_size=-1,
                 num_of_instances_range=(20, 100),
                 prior_filter=None,
                 particles_margin=0.01,
                 estimation_method: EstimationMethod = EstimationMethod.VeryWellSeparated,
                 estimate_noise=False,
                 use_noise_params=True,
                 save_statistics=False,
                 plot=True,
                 save=True,
                 save_dir=os.path.join(ROOT_DIR, f'src/experiments/plots/'),
                 log_level=logging.INFO):
        self._name = name  # Experiment name
        self._estimation_method = estimation_method
        self._data_simulator = simulator
        self._mrc = mrc
        self._filter_basis_size = filter_basis_size
        self._num_of_instances_range = num_of_instances_range
        self._prior_filter = prior_filter
        self._down_sample_size = down_sample_size
        self._estimate_noise = estimate_noise
        self._use_noise_params = use_noise_params
        self._particles_margin = particles_margin
        self._save_statistics = save_statistics

        self._plot = plot
        self._save = save
        self._save_dir = save_dir
        self._log_level = log_level
        self._results = {}
        self.experiment_attr = {}

        curr_date = str(datetime.now().strftime("%d-%m-%Y"))
        curr_time = str(datetime.now().strftime("%H-%M-%S"))
        self.experiment_dir = os.path.join(self._save_dir, curr_date, curr_time)

        if mrc is None:
            if simulator is None:
                simulator = DataSimulator2D()
            self._data = simulator.simulate()

            logger.info(f'Simulating data, number of occurrences is {simulator.num_of_instances}')

            self._noise_std = simulator.noise_std
            self._noise_mean = simulator.noise_mean

            self._signal_length = simulator.signal_length
            self._applied_ctf = simulator.apply_ctf

            self.experiment_attr['clean_data'] = simulator.clean_data
            self.experiment_dir += f'_size_{simulator.signal_length}_std_{self._noise_std}'
        else:
            logger.info(f'Loading given micrograph from {mrc.name}')
            mrc_data = mrc.get_micrograph()
            # self._data = self._data[:min(self._data.shape), :min(self._data.shape)]
            self._data = np.array([mrc_data])
            self._noise_std = mrc.noise_std
            self._noise_mean = mrc.noise_mean
            self._signal_length = None
            self._applied_ctf = True

            self.experiment_dir += f'_{mrc.name}'

        self._number_of_micrographs = self._data.shape[0]
        self._rows = self._data.shape[1]
        self._columns = self._data.shape[2]

        plt.rcParams["figure.figsize"] = (16, 9)
        if self._plot:
            if self._number_of_micrographs > 1:
                cols = 4
                rows = np.ceil(self._number_of_micrographs / cols).astype(int)
                fig, axs = plt.subplots(rows, cols)
                for i in range(self._number_of_micrographs):
                    row = i // cols
                    col = i % cols
                    if rows > 1:
                        axs[row][col].imshow(self._data[i], cmap='gray')
                    else:
                        axs[col].imshow(self._data[i], cmap='gray')
            else:
                plt.imshow(self._data[0], cmap='gray')
            plt.show()

        if signal_length_by_percentage is None:
            signal_length_by_percentage = [3, 4, 5, 6, 8, 10]
        self._signal_length_by_percentage = np.array(signal_length_by_percentage)
        self._sizes_options = np.array(self._rows * self._signal_length_by_percentage // 100, dtype=int)

        if self._save:
            pathlib.Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

        if self._estimation_method == EstimationMethod.Curves:
            logger.info(f'Estimating signal length using Curves method')
            self._length_estimator = LengthEstimator2DCurvesMethod(data=self._data,
                                                                   length_options=self._signal_length_by_percentage,
                                                                   signal_filter_gen_1d=signal_1d_filter_gen,
                                                                   noise_mean=self._noise_mean,
                                                                   noise_std=self._noise_std,
                                                                   logs=self._log_level,
                                                                   experiment_dir=self.experiment_dir)
        else:
            self._length_estimator = LengthEstimator2DVeryWellSeparated(data=self._data,
                                                                        signal_length_by_percentage=self._signal_length_by_percentage,
                                                                        num_of_instances_range=self._num_of_instances_range,
                                                                        prior_filter=self._prior_filter,
                                                                        noise_mean=self._noise_mean,
                                                                        noise_std=self._noise_std,
                                                                        estimate_noise_parameters=self._estimate_noise,
                                                                        use_noise_params=self._use_noise_params,
                                                                        filter_basis_size=self._filter_basis_size,
                                                                        particles_margin=self._particles_margin,
                                                                        save_statistics=self._save_statistics,
                                                                        log_level=self._log_level,
                                                                        plots=self._plot,
                                                                        save=self._save,
                                                                        experiment_dir=self.experiment_dir,
                                                                        experiment_attr=self.experiment_attr)

    def run(self):
        start_time = time.time()
        likelihoods, optimal_coeffs, estimated_signal = self._length_estimator.estimate()
        end_time = time.time()

        self._results = {
            "likelihoods": likelihoods,
            "optimal_coeffs": optimal_coeffs,
            "estimated_signal": estimated_signal,
            "total_time": end_time - start_time,
            "most_likely_size": self._sizes_options[np.nanargmax(likelihoods)]
        }

        self.save_and_plot()

        return self._results

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

        noise_estimation = "Given" if self._use_noise_params else "Learn" if self._estimate_noise else "Estimate"
        if self._mrc is None:
            title += f"Total instances = {self._data_simulator.num_of_instances}, " \
                     f"Num of instances range = {self._num_of_instances_range}, " \
                     f"Basis size = {self._filter_basis_size}\n" \
                     f"Noise estimation = {noise_estimation}, " \
                     f"Particles separation = {self._particles_margin}\n" \
                     f"SNR={self._data_simulator.snr}db (MRC-SNR={self._data_simulator.mrc_snr}db), "
        else:
            title += f"MRC={self._mrc.name}\n"

        title += f"CTF={self._applied_ctf}, " \
                 f"Estimation method={self._estimation_method.name}\n" \
                 f"Most likely length={self._sizes_options[most_likely_index]}, " \
                 f"Took {int(self._results['total_time'])} seconds"
        fig.suptitle(title)

        mrc_fig.imshow(self._data[0], cmap='gray')
        pcm = est_particle_fig.imshow(self._results['estimated_signal'], cmap='gray')
        plt.colorbar(pcm, ax=est_particle_fig)

        most_likely_coeffs = self._results['optimal_coeffs'][most_likely_index]
        est_basis_coeffs_fig.bar(np.arange(len(most_likely_coeffs)), most_likely_coeffs)

        likelihoods_fig.plot(self._sizes_options, likelihoods)
        likelihoods_fig.set_xlabel('Lengths')
        likelihoods_fig.set_ylabel('Likelihood')

        if self._data_simulator:
            pcm = particle_fig.imshow(self._data_simulator.create_signal_instance(), cmap='gray')
            plt.colorbar(pcm, ax=particle_fig)

        if self._save:
            # curr_time = str(datetime.now().strftime("%H-%M-%S"))
            fig_path = os.path.join(self.experiment_dir, f'{self._name}.png')
            plt.savefig(fname=fig_path)
        if self._plot:
            plt.show()

        plt.close()


np.random.seed(504)


def __main__():
    sim_data = DataSimulator2D(rows=500,
                               columns=500,
                               signal_length=30,
                               signal_power=1,
                               # signal_fraction=np.random.randint(10, 20) / 100,
                               # num_of_instances=
                               # method='vws',
                               signal_margin=0,
                               num_of_instances=2,
                               # num_of_instances=np.random.randint(40, 120),
                               number_of_micrographs=1,
                               # signal_gen=Shapes2D.sphere,
                               # signal_gen=lambda l, p: Shapes2D.double_disk(l, l // 2, p, 0),
                               signal_gen=Shapes2D.disk,
                               noise_std=0.0001,
                               noise_mean=0,
                               apply_ctf=False)

    Experiment2D(
        name=f"expy",
        # mrc=Micrograph(file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10128\002.mrc',
        #                downsample=4000,
        #                clip_outliers=True),
        # mrc=Micrograph(file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMD-2984_0010_1000.mrc', clip_outliers=True),
        mrc=Micrograph(
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10049\stack_0250_2x_SumCorr - Copy.mrc',
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10049\stack_0241_2x_SumCorr.mrc',
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10049\stack_0199_2x_SumCorr.mrc',
            file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10081\HCN1apo_0035_2xaligned.mrc',
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10089\TcdA1-0155_frames_sum.mrc',
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10061\EMD-2984_1068.mrc',
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10089\TcdA1-0155_frames_sum.mrc',
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10089\TcdA1-0176_frames_sum.mrc',
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\10028\001.mrc',
            # file_path=r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10061\EMD-2984_0061.mrc',
            downsample=4000,
            low_pass_filter=-1,
            plot=False,
            clip_outliers=True),
        # mrc=Micrograph(file_path=r'C:\Users\tamir\Desktop\Thesis\data\002.mrc', downsample=1000),
        simulator=sim_data,
        # signal_length_by_percentage=[5, 7, 8, 10],
        signal_length_by_percentage=[2, 3, 4, 5, 6.5, 8],
        # signal_length_by_percentage=[2, 6, 10],
        # signal_length_by_percentage=[6],
        # signal_length_by_percentage=[5, 8, 10, 12],
        num_of_instances_range=(30, 30),
        # num_of_instances_range=(2, 2),
        # num_of_instances_range=(int(sim_data.num_of_instances * 0.7), int(sim_data.num_of_instances * 0.7)),
        # down_sample_size=500,
        use_noise_params=False,
        estimate_noise=False,
        filter_basis_size=5,
        save_statistics=True,
        particles_margin=0,
        plot=True,
        save=True,
        log_level=logging.INFO
    ).run()


if __name__ == '__main__':
    __main__()
