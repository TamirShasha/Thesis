import numpy as np
import os

from src.constants import ROOT_DIR
from src.experiments.experiments_1d import Experiment
from src.experiments.experiments_2d import Experiment2D, EstimationMethod, SignalPowerEstimator
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D


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

    Experiment2D(
        name=exp_name,
        simulator=sim_data,
        estimation_method=method,
        signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
        length_options=np.arange(50, 500, 10),
        signal_num_of_occurrences_boundaries=(20, 200),
        signal_area_coverage_boundaries=(0.05, 0.3),
        num_of_power_options=10,
        plot=False,
        save=True,
        save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/')
    ).run()


def ellipse23(d, p):
    return Shapes2D.ellipse(d, d // 1.5, p)


def ellipse12(d, p):
    return Shapes2D.ellipse(d, d // 2, p)


"""
1D Experiments
"""
run_1d_baseline_experiment('1D_10k_300_3std', 100000, 3)
run_1d_baseline_experiment('1D_30k_300_5std', 30000, 5)
run_1d_baseline_experiment('1D_50k_300_8std', 50000, 8)
run_1d_baseline_experiment('1D_100k_300_10std', 100000, 10)
run_1d_baseline_experiment('1D_200k_300_12std', 200000, 12)
run_1d_baseline_experiment('1D_400k_300_12std', 400000, 15)

"""
2D Disk Experiments with noise std 3,5,8,10,12,15
"""
run_2d_baseline_experiment('2D_4000x4000_disk_300_curves_3std', Shapes2D.disk, 3, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_disk_300_vws_3std', Shapes2D.disk, 3, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_disk_300_curves_5std', Shapes2D.disk, 5, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_disk_300_vws_5std', Shapes2D.disk, 5, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_disk_300_curves_8std', Shapes2D.disk, 8, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_disk_300_vws_8std', Shapes2D.disk, 8, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_disk_300_curves_10std', Shapes2D.disk, 10, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_disk_300_vws_10std', Shapes2D.disk, 10, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_disk_300_curves_12std', Shapes2D.disk, 12, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_disk_300_vws_12std', Shapes2D.disk, 12, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_disk_300_curves_15std', Shapes2D.disk, 15, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_disk_300_vws_15std', Shapes2D.disk, 15, EstimationMethod.WellSeparation)

"""
2D Ellipse 2:3 Experiments with noise std 3,5,8,10,12,15
"""

run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_curves_3std', ellipse23, 3, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_vws_3std', ellipse23, 3, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_curves_5std', ellipse23, 5, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_vws_5std', ellipse23, 5, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_curves_8std', ellipse23, 8, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_vws_8std', ellipse23, 8, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_curves_10std', ellipse23, 10, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_vws_10std', ellipse23, 10, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_curves_12std', ellipse23, 12, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_vws_12std', ellipse23, 12, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_curves_15std', ellipse23, 15, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse23_300_vws_15std', ellipse23, 15, EstimationMethod.WellSeparation)

"""
2D Ellipse 1:2 Experiments with noise std 3,5,8,10,12,15
"""

run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_curves_3std', ellipse12, 3, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_vws_3std', ellipse12, 3, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_curves_5std', ellipse12, 5, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_vws_5std', ellipse12, 5, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_curves_8std', ellipse12, 8, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_vws_8std', ellipse12, 8, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_curves_10std', ellipse12, 10, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_vws_10std', ellipse12, 10, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_curves_12std', ellipse12, 12, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_vws_12std', ellipse12, 12, EstimationMethod.WellSeparation)

run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_curves_15std', ellipse12, 15, EstimationMethod.Curves)
run_2d_baseline_experiment('2D_4000x4000_ellipse12_300_vws_15std', ellipse12, 15, EstimationMethod.WellSeparation)
