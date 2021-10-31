import os

from src.constants import ROOT_DIR
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.experiments.experiments_2d import Experiment2D, EstimationMethod

NOISE_MEAN = 0
NOISE_STD = 8
LENGTH_OPTIONS = [80, 100, 120, 140]
SIGNAL_SHAPES = [(Shapes2D.disk, 'disk'),
                 (Shapes2D.sphere, 'sphere')]
sizes = [100, 120]

for i, (signal_shape, shape_name) in enumerate(SIGNAL_SHAPES):
    for signal_size in sizes:
        data_simulator = DataSimulator2D(rows=1000,
                                         columns=1000,
                                         signal_length=signal_size,
                                         signal_power=1,
                                         signal_fraction=1 / 6,
                                         signal_gen=signal_shape,
                                         noise_std=NOISE_STD,
                                         noise_mean=NOISE_MEAN,
                                         apply_ctf=False)

        Experiment2D(
            name=f"{shape_name}_{signal_size}",
            simulator=data_simulator,
            estimation_method=EstimationMethod.VeryWellSeparated,
            filter_basis_size=10,
            fixed_num_of_occurrences=30,
            length_options=LENGTH_OPTIONS,
            plot=False,
            save=True,
            save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/k_margin/')
        ).run()
