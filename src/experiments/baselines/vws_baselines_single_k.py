import os
from datetime import datetime
import numpy as np
from multiprocessing import Process

from src.constants import ROOT_DIR
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.experiments.experiments_2d import Experiment2D, EstimationMethod

NOISE_MEAN = 0
NOISE_STD = 8
N = 1000
LENGTH_OPTIONS_PERC = np.array([2, 2.5, 3, 4, 5, 6, 8, 10])
SIGNAL_SHAPES = [(Shapes2D.disk, 'disk'),
                 (Shapes2D.sphere, 'sphere'),
                 (lambda l, p: Shapes2D.ellipse(l, l // 1.7, p), 'ellipse'),
                 (lambda l, p: Shapes2D.double_disk(l, l // 2, p, 0), 'ring')]
sizes = np.array([2.5, 3, 4, 5, 6, 8]) * N / 100

now_str = datetime.now().strftime("%H_%M_%Y_%m_%d")

for i, (signal_shape, shape_name) in enumerate(SIGNAL_SHAPES):
    for signal_size in sizes:
        data = DataSimulator2D(rows=N,
                               columns=N,
                               signal_length=int(signal_size),
                               signal_power=1,
                               signal_fraction=1 / 5,
                               signal_margin=0,
                               # num_of_instances=np.random.randint(80, 120),
                               signal_gen=signal_shape,
                               noise_std=NOISE_STD,
                               noise_mean=NOISE_MEAN,
                               apply_ctf=False)

        name = f"{shape_name}_{int(signal_size)}"
        experiment = Experiment2D(
            name=name,
            simulator=data,
            estimation_method=EstimationMethod.VeryWellSeparated,
            signal_length_by_percentage=LENGTH_OPTIONS_PERC,
            num_of_instances_range=(int(data.num_of_instances * 0.7), int(data.num_of_instances * 0.7)),
            estimate_noise=False,
            use_noise_params=True,
            filter_basis_size=7,
            particles_margin=0,
            save_statistics=True,
            plot=False,
            save=True,
            save_dir=os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/given_k/{now_str}/{name}/')
        )
        # exit()

        Process(target=experiment.run).start()
