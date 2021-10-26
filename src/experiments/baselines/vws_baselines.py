from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.experiments.experiments_2d import Experiment2D, EstimationMethod

TIMES = 3
NOISE_MEAN = 0
NOISE_STD = 10
LENGTH_OPTIONS = [20, 40, 60, 80, 100, 120, 140, 160]
SIGNAL_SHAPES = [Shapes2D.disk,
                 Shapes2D.sphere,
                 lambda l, p: Shapes2D.ellipse(l, l // 1.7, p),
                 lambda l, p: Shapes2D.double_disk(l, l // 2, p, 0)]
sizes = [40, 80, 120]

for i, signal_shape in enumerate(SIGNAL_SHAPES):
    print(f'At signal shape #{i + 1}')
    for signal_size in sizes:
        data = DataSimulator2D(rows=1000,
                               columns=1000,
                               signal_length=signal_size,
                               signal_power=1,
                               signal_fraction=1 / 6,
                               signal_gen=signal_shape,
                               noise_std=NOISE_STD,
                               noise_mean=NOISE_MEAN,
                               apply_ctf=False)

        Experiment2D(
            name=f"baseline_disk",
            simulator=data,
            estimation_method=EstimationMethod.VeryWellSeparated,
            length_options=LENGTH_OPTIONS,
            plot=False,
            save=True
        ).run()
