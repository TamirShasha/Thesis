import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from src.utils.utils import add_gaus_noise

n = 1000
d = 20
p = 1
k = 20
noise_std = 1
seperation = 5


def arange_data(n, d, p, k, noise_std, seperation):
    signal = np.array([p] * d + [0] * seperation)
    num_of_zeros = n - len(signal) * k
    if num_of_zeros < k:
        print('bad params')

    signals_locations = np.sort(np.random.randint(0, num_of_zeros, k))

    pulses = np.zeros(n)
    for i, location in enumerate(signals_locations):
        loc = i * len(signal) + location
        pulses[loc: loc + len(signal)] = signal

    y = add_gaus_noise(pulses, 0, noise_std)
    return y, pulses


data, pulses = arange_data(n, d, p, k, noise_std, seperation)

d_arr = [1, 5, 10, 15, 30, 50, 100]

plt.figure()
plt.plot(data)
plt.show()

d_probs = []
for d in d_arr:
    probs = []
    correlation_length = d + 2 * seperation
    for i in np.arange(seperation, n - d - seperation):
        left = np.log(np.prod(np.exp(-(data[i - seperation:i]) ** 2 / (noise_std ** 2))))
        middle = np.log(np.prod(np.exp(-(data[i:i + d] - p) ** 2 / (noise_std ** 2))))
        right = np.log(np.prod(np.exp(-(data[i + d:i + d + seperation]) ** 2 / (noise_std ** 2))))
        prob = (left + middle + right) / correlation_length
        probs.append(prob)

    probs = np.array(probs)
    peaks, _ = find_peaks(probs, distance=correlation_length - 1)
    top_k_peaks = peaks[np.argsort(probs[peaks])][-10:]
    print(f'for d={d}, value is: {np.prod(probs[top_k_peaks])}')

    # plt.plot(probs)
    # plt.plot(peaks, probs[peaks], "x")
    # plt.plot(np.zeros_like(probs), "--", color="gray")
    # plt.show()

    d_probs.append(probs)
