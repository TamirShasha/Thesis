import matplotlib.pyplot as plt
import numpy as np

from src.experiments.data_simulator_1d import DataSimulator1D

pulses = lambda d: np.ones(d)

font = {'size': 14}
import matplotlib
matplotlib.rc('font', **font)

np.random.seed(511)
noise_std = 10
sim_data = DataSimulator1D(size=3000,
                           signal_size=70,
                           # signal_fraction=0.1,
                           signal_margin=0.0001,
                           num_of_instances=10,
                           signal_gen=pulses,
                           noise_std=noise_std,
                           noise_mean=0)

noisy_data = sim_data.simulate()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(noisy_data, label='noisy data')
ax.plot(sim_data.clean_data, label='underlying signals')
ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(fr'C:\Users\tamir\Desktop\New folder\Noisy_Data_{str(noise_std).replace(".", "_")}.png')
plt.close()
# plt.show()
