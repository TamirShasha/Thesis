import numpy as np
import matplotlib.pyplot as plt

from src.utils import arange_data

n = 2000
d = 20
p = 1
k = 20
noise_std = 2.5

data, pulses = arange_data(n, d, p, k, noise_std)

d_arr = np.arange(5, 50, 5)

plt.figure()
plt.plot(data)
plt.show()

d_probs = []
for d in d_arr:
    probs = []
    for i in np.arange(n - d):
        prob = np.prod(np.exp(-(data[i:i + d] - p) ** 2 / (noise_std ** 2)))
        probs.append(prob)
    plt.figure()
    plt.title(d)
    plt.plot(pulses * np.max(probs))
    plt.plot(probs)
    plt.show()

    d_probs.append(probs)
