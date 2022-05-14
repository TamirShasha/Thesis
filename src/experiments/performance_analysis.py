import os
import numpy as np
import pandas as pd
import warnings

from src.constants import ROOT_DIR
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.experiments.experiments_2d import Experiment2D, EstimationMethod

np.random.seed(500)
warnings.simplefilter("ignore")

N = 500
NOISE_MEAN = 0
PARTICLES_MARGIN = [0, 0.02]
NOISE_STDS = [0.1, 1, 1.5, 3, 5]
SIZES = np.array([2, 3, 5, 8], dtype=int) * (N // 100)
NUM_OF_INSTANCES_RANGE = (30, 30)
FILTER_BASIS_SIZE = 2
NOISE_OPTIONS = [[True, False], [False, False]]
DENSITY = [0.25, 0.2, 0.15, 0.12, 0.1]

POSSIBLE_SIZES = np.array([1.5, 2, 3, 5, 8, 10, 12], dtype=int) * (N // 100)


def run_and_save_experiments():
    # results_df = pd.DataFrame()
    results_df = pd.read_csv('performance.csv')

    for noise_std in [2, 2.5]:
        for noise_options in NOISE_OPTIONS:
            for frac in DENSITY:
                for particle_margin in PARTICLES_MARGIN:
                    for signal_size in SIZES:
                        data_simulator = DataSimulator2D(rows=N,
                                                         columns=N,
                                                         signal_length=signal_size,
                                                         signal_power=1,
                                                         signal_fraction=frac,
                                                         signal_gen=Shapes2D.sphere,
                                                         signal_margin=particle_margin,
                                                         noise_std=noise_std,
                                                         noise_mean=NOISE_MEAN,
                                                         apply_ctf=False)

                        result = Experiment2D(
                            name=f"sphere_{signal_size}",
                            simulator=data_simulator,
                            signal_length_by_percentage=[1.5, 2, 3, 5, 8, 10, 12],
                            estimation_method=EstimationMethod.VeryWellSeparated,
                            filter_basis_size=FILTER_BASIS_SIZE,
                            num_of_instances_range=NUM_OF_INSTANCES_RANGE,
                            use_noise_params=noise_options[0],
                            estimate_noise=noise_options[1],
                            save_statistics=True,
                            particles_margin=particle_margin,
                            plot=False,
                            save=True,
                            save_dir=os.path.join(ROOT_DIR, f'src/experiments/performance_analysis_plots/')
                        ).run()

                        result_row = {
                            "signal_size": signal_size,
                            "num_of_instances_range": NUM_OF_INSTANCES_RANGE,
                            "use_noise_params": noise_options[0],
                            "estimate_noise": noise_options[1],
                            "signals_density": frac,
                            "noise_mean": NOISE_MEAN,
                            "noise_std": noise_std,
                            "most_likely_size": result['most_likely_size'],
                            "optimal_coeffs": result['optimal_coeffs'],
                            "likelihoods": result['likelihoods'],
                            "particle_margin": particle_margin
                        }

                        results_df = results_df.append(result_row, ignore_index=True)
                        results_df.to_csv('performance.csv')


import matplotlib.pyplot as plt


def plot_results(use_elimination_method=True):
    df = pd.read_csv('performance.csv')



    for noise_std in df['noise_std'].unique():
    # for noise_std in [2]:
        for noise_options in NOISE_OPTIONS:
            for particle_margin in [0, 0.02]:

                df_reduced = df[(df.particle_margin == particle_margin) &
                                (df.noise_std == noise_std) &
                                (df.use_noise_params == noise_options[0]) &
                                (df.estimate_noise == noise_options[1])][
                    ['signal_size', 'signals_density', 'most_likely_size', 'likelihoods', 'optimal_coeffs']]

                img = np.zeros(shape=(len(DENSITY), len(SIZES), 3))

                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.matshow(img, interpolation='nearest')

                for i, size in enumerate(SIZES):
                    for j, density in enumerate(DENSITY):
                        entry = df_reduced[
                            (df_reduced.signal_size == size) & (df_reduced.signals_density == density)].values[
                            0]
                        likelihoods = np.fromstring(entry[3].replace('\n', '')[1:-1], sep=' ')
                        coeffs = np.array([np.fromstring(s.strip()[1:-1], sep=' ') for s in entry[4][1:-1].split('\n')])

                        if use_elimination_method:
                            estimated_size = POSSIBLE_SIZES[np.nanargmax(likelihoods[np.where(coeffs[:, 1] < 0)])]
                        else:
                            estimated_size = entry[2]
                        if size == estimated_size:
                            img[j][i] = [69, 139, 116]
                        elif np.abs(size - estimated_size) <= 10:
                            img[j][i] = [255, 99, 71]
                        else:
                            img[j][i] = [255, 64, 64]

                        ax.text(i, j, str(estimated_size), ha='center', va='center')
                img /= 255

                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # cax = ax.matshow(img, interpolation='nearest')
                ax.matshow(img, interpolation='nearest')
                ax.set_title('Predicted Sizes\n'
                             f'Noise STD = {noise_std}, Particles Margin = {particle_margin}\n'
                             f'{"Given noise params" if noise_options[0] else "Estimate noise params"}')

                # for i, size in enumerate(SIZES):
                #     for j, density in enumerate(DENSITY):
                #         estimated_size = \
                #             df_reduced[
                #                 (df_reduced.signal_size == size) & (df_reduced.signals_density == density)].values[
                #                 0][2]
                #         ax.text(i, j, str(estimated_size), ha='center', va='center')

                ax.set_xlabel('True size')
                ax.set_ylabel('Density')


                ax.set_xticklabels([''] + [str(s) for s in SIZES])
                ax.set_yticklabels([''] + [str(d) for d in DENSITY])

                file_name = f'std_{noise_std}'
                file_name += '_pm'
                file_name += '0' if particle_margin == 0 else '002'
                file_name += '_given' if noise_options[0] else 'estimated'
                fig_path = os.path.join(f'to_save/{file_name}.png')
                plt.savefig(fname=fig_path)
                # plt.show()
                plt.close()


# run_and_save_experiments()
plot_results()
