import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from numpy.fft import fft2, fftshift

# Define the parameters for the Fourier-Bessel transform
R = 50 # maximum radius
N = 101 # number of samples
K = 25 # number of basis elements

# Compute the sampling points and the Bessel function values
r = np.linspace(0, R, N)
j = jn(np.arange(K), r[:, np.newaxis])

# Compute the Fourier transform of the Bessel function values
Fj = fftshift(fft2(j))

# Display the basis elements as images
fig, axs = plt.subplots(K, 1, figsize=(8, 2*K))
for k in range(K):
    axs[k].imshow(np.abs(Fj[:, :, k]), extent=(-R, R, -R, R))
    axs[k].set_title(f'Basis {k}')
    axs[k].axis('off')
plt.tight_layout()
plt.show()