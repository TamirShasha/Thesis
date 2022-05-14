import os
import numpy as np

from src.constants import ROOT_DIR
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.experiments.experiments_2d import Experiment2D, EstimationMethod

np.random.seed(500)

NOISE_MEAN = 0
NOISE_STD = 5
sizes = [30, 40, 50, 80, 100]

for noise_options in [[True, False], [False, False]]:
    for frac in [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]:
        for signal_size in sizes:
            data_simulator = DataSimulator2D(rows=1000,
                                             columns=1000,
                                             signal_length=signal_size,
                                             signal_power=1,
                                             signal_fraction=frac,
                                             signal_gen=Shapes2D.sphere,
                                             signal_margin=0,
                                             noise_std=NOISE_STD,
                                             noise_mean=NOISE_MEAN,
                                             apply_ctf=False)

            Experiment2D(
                name=f"sphere_{signal_size}",
                simulator=data_simulator,
                signal_length_by_percentage=[1.5, 2, 3, 5, 8, 10, 12],
                estimation_method=EstimationMethod.VeryWellSeparated,
                filter_basis_size=2,
                num_of_instances_range=(70, 70),
                use_noise_params=noise_options[0],
                estimate_noise=noise_options[1],
                save_statistics=True,
                particles_margin=0,
                plot=False,
                save=True,
                save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/05_04/')
            ).run()
