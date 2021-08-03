import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pathlib

from src.constants import ROOT_DIR
from src.algorithms.length_estimator_2d_curves_method import LengthEstimator2DCurvesMethod
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D

TIMES = 30
EXP_LENGTH_OPTIONS = np.concatenate([[20, 50, 70], np.arange(100, 701, 100)])
LENGTH_OPTIONS = np.concatenate([[20, 50, 70], np.arange(100, 1001, 100)])

NOISE_STD = 20

errors = np.zeros_like(EXP_LENGTH_OPTIONS)
for i, length in enumerate(EXP_LENGTH_OPTIONS):
    acc_results = 0
    for t in range(TIMES):
        print(f'At experiment #{t + 1} for length {length}')

        data = DataSimulator2D(rows=4000,
                               columns=4000,
                               signal_length=length,
                               signal_power=1,
                               signal_fraction=1 / 6,
                               signal_gen=Shapes2D.sphere,
                               noise_std=NOISE_STD,
                               noise_mean=0,
                               apply_ctf=False).simulate()

        length_estimator = LengthEstimator2DCurvesMethod(data=data,
                                                         length_options=LENGTH_OPTIONS,
                                                         signal_filter_gen_1d=lambda d, p=1: np.full(d, p),
                                                         noise_mean=0,
                                                         noise_std=NOISE_STD,
                                                         logs=False)

        likelihoods, most_likely_length, most_likely_power = length_estimator.estimate()
        acc_results += most_likely_length

    errors[i] = np.abs(acc_results / TIMES - length)

plt.plot(EXP_LENGTH_OPTIONS, errors)
plt.xlabel('Length Options')
plt.ylabel('Relative Error')

save_dir = os.path.join(ROOT_DIR, f'src/experiments/baselines/plots/')
curr_date = str(datetime.now().strftime("%d-%m-%Y"))
dir_path = os.path.join(save_dir, curr_date)
pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

curr_time = str(datetime.now().strftime("%H-%M-%S"))
fig_path = os.path.join(dir_path, f'{curr_time}_size_experiment.png')
plt.savefig(fname=fig_path)

plt.show()
plt.close()
