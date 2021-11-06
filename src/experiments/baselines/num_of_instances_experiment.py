import os

from src.constants import ROOT_DIR
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.experiments.experiments_2d import Experiment2D, EstimationMethod

NOISE_MEAN = 0
NOISE_STD = 8
LENGTH_OPTIONS = [20, 40, 60, 80, 100, 120, 140, 160]
SIGNAL_SHAPES = [(Shapes2D.disk, 'disk'),
                 (Shapes2D.sphere, 'sphere'),
                 (lambda l, p: Shapes2D.ellipse(l, l // 1.7, p), 'ellipse'),
                 (lambda l, p: Shapes2D.double_disk(l, l // 2, p, 0), 'ring')]
sizes = [40, 60, 80, 100, 120]

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
            estimate_noise=True,
            length_options=LENGTH_OPTIONS,
            num_of_instances_range=int(0.8 * data_simulator.occurrences),
            plot=False,
            save=True,
            save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/fixed_num_of_instances/')
        ).run()



