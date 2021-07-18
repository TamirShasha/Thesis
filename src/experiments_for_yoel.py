import numpy as np
import matplotlib.pyplot as plt

from src.experiments.experiments_1d import Experiment as Experiment1D
from src.experiments.experiments_2d import Experiment2D
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.algorithms.length_estimator_1d import SignalPowerEstimator
from src.algorithms.length_estimator_2d import EstimationMethod
from src.experiments.micrograph import MICROGRAPHS
from src.experiments.particles_projections import PARTICLE_200

np.random.seed(500)  # for the same figures

# Experiment1D(
#     name="first_one_dim_exp",
#     n=40000,
#     d=200,
#     length_options=None,  # as None, it will auto choose np.arange(self._d // 4, int(self._d * 3), 10)
#     signal_fraction=1 / 5,  # means that all of signal instances together covers 1/5 of the area, i.e. n
#     signal_fn=lambda d: np.full(d, 1),  # the signal that is generated (only for the data generation)
#     signal_filter_gen=lambda d: np.full(d, 1),  # the signal filter for the algorithm
#     signal_power_estimator_method=SignalPowerEstimator.FirstMoment,  # the method for estimating total signal power
#     noise_std=10,
#     noise_mean=0,
#     plot=True,
#     save=True  # will save the results at src/experiments/plots/
# ).run()
#
#
# def signal_fn_second_exp(d):
#     return (1 + np.sin(30 * np.linspace(0, 1, d)) / 7) * (1 - 0.5 * np.square(np.linspace(-1, 1, d)))
#
#
# plt.plot(signal_fn_second_exp(200))
# plt.show()
# Experiment1D(
#     name="second_one_dim_exp",
#     n=100000,
#     d=200,
#     length_options=None,  # as None, it will auto choose np.arange(self._d // 4, int(self._d * 3), 10)
#     signal_fraction=1 / 5,  # means that all of signal instances together covers 1/5 of the area, i.e. n
#     signal_fn=signal_fn_second_exp,  # the signal that is generated (only for the data generation)
#     signal_filter_gen=lambda d: np.full(d, 1),  # the signal filter for the algorithm
#     signal_power_estimator_method=SignalPowerEstimator.FirstMoment,  # the method for estimating total signal power
#     noise_std=8,
#     noise_mean=0,
#     plot=True,
#     save=True  # will save the results at src/experiments/plots/
# ).run()
#
sim_data = DataSimulator2D(rows=4000,
                           columns=4000,
                           signal_length=200,  # signal 'diamiter'
                           signal_power=1,
                           signal_fraction=1 / 6,
                           signal_gen=Shapes2D.sphere,
                           noise_std=20,
                           noise_mean=0,
                           apply_ctf=False)
Experiment2D(
    name=f"simulated_exp_two_dim",
    simulator=sim_data,
    estimation_method=EstimationMethod.Curves,  # using the curves method or the well-separation method for estimator
    signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
    length_options=np.arange(10, 500, 20),
    signal_num_of_occurrences_boundaries=(0, 20000),  # no need to touch, does nothing at the moment..
    signal_area_coverage_boundaries=(0.05, 0.20),  # no need to touch, does nothing at the moment..
    num_of_power_options=10,  # the amount of power options the algorithm will try, usually 10 is enough.
    plot=True,
    save=True
).run()

# sim_data = DataSimulator2D(rows=4000,
#                            columns=4000,
#                            signal_length=PARTICLE_200.particle_length,  # signal 'diamiter'
#                            signal_power=1,
#                            signal_fraction=1 / 6,
#                            signal_gen=PARTICLE_200.get_signal_gen(),
#                            noise_std=15,
#                            noise_mean=0,
#                            apply_ctf=False)
#
# Experiment2D(
#     name=f"simulated_exp_two_dim",
#     simulator=sim_data,
#     estimation_method=EstimationMethod.Curves,  # using the curves method or the well-separation method for estimator
#     signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
#     length_options=np.arange(10, 500, 20),
#     signal_num_of_occurrences_boundaries=(0, 20000),  # no need to touch, does nothing at the moment..
#     signal_area_coverage_boundaries=(0.05, 0.20),  # no need to touch, does nothing at the moment..
#     num_of_power_options=10,  # the amount of power options the algorithm will try, usually 10 is enough.
#     plot=True,
#     save=True
# ).run()


#
# Experiment2D(
#     name=f"real_mrc_exp_two_dim",
#     mrc=MICROGRAPHS['002_normalized'],
#     estimation_method=EstimationMethod.Curves,
#     signal_power_estimator_method=SignalPowerEstimator.FirstMoment,
#     length_options=np.arange(10, 500, 20),
#     signal_num_of_occurrences_boundaries=(0, 20000),
#     signal_area_coverage_boundaries=(0.05, 0.20),
#     num_of_power_options=10,
#     plot=True,
#     save=True
# ).run()
