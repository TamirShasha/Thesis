import numpy as np
import os

from src.constants import ROOT_DIR
from src.experiments.experiments_1d import Experiment
from src.experiments.experiments_2d import Experiment2D, EstimationMethod, SignalPowerEstimator
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D


def one_dim_experiments():
    def run_1d_baseline_experiment(exp_name, n, noise_std):
        print('#### START NEW EXPERIMENT ####')
        Experiment(
            name=exp_name,
            n=n,
            d=300,
            signal_fraction=1 / 5,
            length_options=np.arange(50, 500, 10),
            signal_fn=lambda d: np.full(d, 1),
            signal_filter_gen=lambda d: np.full(d, 1),
            noise_std=noise_std,
            signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
            plot=False,
            save=True,
            save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/')
        ).run()
        print('#### DONE EXPERIMENT ####')

        """
        1D Experiments
        """
    # run_1d_baseline_experiment('1D_10k_300_3std', 300000, 3)
    # run_1d_baseline_experiment('1D_30k_300_5std', 400000, 5)
    # run_1d_baseline_experiment('1D_50k_300_8std', 500000, 8)
    # run_1d_baseline_experiment('1D_100k_300_10std', 600000, 10)
    # run_1d_baseline_experiment('1D_200k_300_12std', 700000, 12)
    # run_1d_baseline_experiment('1D_400k_300_12std', 800000, 15)


# one_dim_experiments()

def run_2d_baseline_experiment(exp_name, signal_gen, noise_std, method):
    """
    baseline
    """
    sim_data = DataSimulator2D(rows=4000,
                               columns=4000,
                               signal_length=300,
                               signal_power=1,
                               signal_fraction=1 / 5,
                               signal_gen=signal_gen,
                               noise_std=noise_std,
                               noise_mean=0)

    print('#### START NEW EXPERIMENT ####')
    Experiment2D(
        name=exp_name,
        simulator=sim_data,
        estimation_method=method,
        signal_power_estimator_method=SignalPowerEstimator.SecondMoment,
        signal_length_by_percentage=np.arange(50, 500, 10),
        signal_num_of_occurrences_boundaries=(20, 200),
        signal_area_coverage_boundaries=(0.05, 0.3),
        num_of_power_options=10,
        plot=False,
        save=True,
        save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/')
    ).run()
    print('#### DONE EXPERIMENT ####')


def ellipse23(d, p):
    return Shapes2D.ellipse(d, d // 1.5, p)


def ellipse12(d, p):
    return Shapes2D.ellipse(d, d // 2, p)


# for std in [3, 5, 8, 10, 12, 15]:
#     for signal_gen in [Shapes2D.disk, ellipse23, ellipse12]:
#         run_2d_baseline_experiment(f'2D_4000x4000_{signal_gen.__name__}_300_curves_{std}std', signal_gen, std,
#                                    EstimationMethod.Curves)
#         run_2d_baseline_experiment(f'2D_4000x4000_{signal_gen.__name__}_300_vws_{std}std', signal_gen, std,
#                                    EstimationMethod.WellSeparation)

"""
Shpere Experiments
"""


def run_2d_baseline_sphere_experiment(exp_name, noise_std, method):
    """
    baseline
    """
    sim_data = DataSimulator2D(rows=4000,
                               columns=4000,
                               signal_length=300,
                               signal_power=1,
                               signal_fraction=1 / 8,
                               signal_gen=Shapes2D.sphere,
                               noise_std=noise_std,
                               noise_mean=0)

    print('#### START NEW EXPERIMENT ####')
    Experiment2D(
        name=exp_name,
        simulator=sim_data,
        estimation_method=method,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        signal_length_by_percentage=np.arange(50, 500, 10),
        signal_num_of_occurrences_boundaries=(20, 200),
        signal_area_coverage_boundaries=(0.05, 0.3),
        num_of_power_options=10,
        plot=False,
        save=True,
        save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/')
    ).run()
    print('#### DONE EXPERIMENT ####')


# for std in [3, 5, 8, 10, 12, 15]:
#     run_2d_baseline_sphere_experiment(f'2D_4000x4000_sphere_300_curves_{std}std', std, EstimationMethod.Curves)
#     run_2d_baseline_sphere_experiment(f'2D_4000x4000_sphere_300_vws_{std}std', std, EstimationMethod.WellSeparation)


def run_2d_baseline_sphere_ctf_experiment(exp_name, noise_std, method):
    """
    baseline
    """
    sim_data = DataSimulator2D(rows=4000,
                               columns=4000,
                               signal_length=300,
                               signal_power=10,
                               signal_fraction=1 / 8,
                               signal_gen=Shapes2D.sphere,
                               noise_std=noise_std,
                               noise_mean=0,
                               apply_ctf=True)

    print('#### START NEW EXPERIMENT ####')
    Experiment2D(
        name=exp_name,
        simulator=sim_data,
        estimation_method=method,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        signal_length_by_percentage=np.arange(50, 500, 10),
        signal_num_of_occurrences_boundaries=(20, 200),
        signal_area_coverage_boundaries=(0.05, 0.3),
        num_of_power_options=10,
        plot=False,
        save=True,
        save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/')
    ).run()
    print('#### DONE EXPERIMENT ####')


# for std in [3, 5, 8, 10, 12, 15]:
#     run_2d_baseline_sphere_ctf_experiment(f'2D_4000x4000_sphere_300_curves_{std}std', std, EstimationMethod.Curves)
#     run_2d_baseline_sphere_ctf_experiment(f'2D_4000x4000_sphere_300_vws_{std}std', std, EstimationMethod.WellSeparation)
