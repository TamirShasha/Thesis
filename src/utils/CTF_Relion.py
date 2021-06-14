import numpy as np


def apply_CTF(image, CTF):
    return


def cryo_CTF_Relion(square_side, pixel_size, defocus_u, defocus_v, defocus_angle, spherical_aberration, amplitude_contrast, voltage=300):
    """
        Compute the contrast transfer function corresponding an n x n image with
        the sampling interval DetectorPixelSize.

    """
    #  wavelength in nm
    wave_length = voltage_to_wavelength(voltage)

    # Divide by 10 to make pixel size in nm. BW is the bandwidth of
    #  the signal corresponding to the given pixel size
    bw = 1 / (pixel_size / 10)

    s, theta = radius_norm(square_side, origin=fctr(square_side))

    # RadiusNorm returns radii such that when multiplied by the
    #  bandwidth of the signal, we get the correct radial frequnecies
    #  corresponding to each pixel in our nxn grid.
    s = s * bw

    DFavg = (defocus_u + defocus_v) / 2
    DFdiff = (defocus_u - defocus_v)
    df = DFavg + DFdiff * np.cos(2 * (theta - defocus_angle)) / 2
    k2 = np.pi * wave_length * df
    # 10**6 converts spherical_aberration from mm to nm
    k4 = np.pi / 2*10**6 * spherical_aberration * wave_length**3
    chi = k4 * s**4 - k2 * s**2

    return np.sqrt(1 - amplitude_contrast ** 2) * np.sin(chi) - amplitude_contrast * np.cos(chi)


def radius_norm(n: int, origin=None):
    """
        Create an n(1) x n(2) array where the value at (x,y) is the distance from the
        origin, normalized such that a distance equal to the width or height of
        the array = 1.  This is the appropriate function to define frequencies
        for the fft of a rectangular image.

        For a square array of size n (or [n n]) the following is true:
        RadiusNorm(n) = Radius(n)/n.
        The org argument is optional, in which case the FFT center is used.

        Theta is the angle in radians.

        (Transalted from Matlab RadiusNorm.m)
    """

    if isinstance(n, int):
        n = np.array([n, n])

    if origin is None:
        origin = np.ceil((n + 1) / 2)

    a, b = origin[0], origin[1]
    y, x = np.meshgrid(np.arange(1-a, n[0]-a+1)/n[0],
                       np.arange(1-b, n[1]-b+1)/n[1])  # zero at x,y
    radius = np.sqrt(x ** 2 + y ** 2)

    theta = np.arctan2(x, y)

    return radius, theta


def fctr(n):
    """ Center of an FFT-shifted image. We use this center
        coordinate for all rotations and centering operations. """

    if isinstance(n, int):
        n = np.array([n, n])

    return np.ceil((n + 1) / 2)


def voltage_to_wavelength(voltage, version='aspire'):
    if version == 'aspire':
        # aspire matlab version
        wave_length = 1.22639 / np.sqrt(voltage * 1000 + 0.97845 * voltage ** 2)
    elif version == 'cov3d':
        # cov3d matlab version
        wave_length = 1.22643247 / np.sqrt(voltage * 1000 + 0.978466 * voltage ** 2)
    else:
        raise ValueError('version can only be aspire or cov3d, but got {} instead'.format(version))
    return wave_length


## Example
# import matplotlib.pyplot as plt
# pixel_size = 1.3399950228756292
# DefocusU = 2334.4699219
# DefocusV = 2344.5949219
# DefocusAngle = 0.6405358529352114
# spherical_aberration = 2.0
# amplitude_contrast = 0.1
#
# CTF = cryo_CTF_Relion(300, pixel_size, DefocusU, DefocusV, DefocusAngle, spherical_aberration, amplitude_contrast)
#
# plt.imshow(CTF)
# plt.show()
