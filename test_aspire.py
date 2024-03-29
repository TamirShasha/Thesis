if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from aspire.basis import FFBBasis2D
    from aspire.

    b = FFBBasis2D(size=(50, 50), dtype=float)
    print(b)
    b.evaluate()


# import aspire.utils.common as common
import scipy.special as sp
import numpy as np
# import finufftpy
# from aspire.utils.constants import BESSEL_NPY


def compute_spca(images, noise_v_r, adaptive_support=False):
    num_images = images.shape[2]
    resolution = images.shape[0]

    if adaptive_support:
        raise NotImplementedError('Adaptive support was not implemented yet')
        # energy_thresh = 0.99
        #
        # # Estimate bandlimit and compact support size
        # [bandlimit, support_size] = choose_support_v6(common.fast_cfft2(images), energy_thresh)
        # # Rescale between 0 and 0.5
        # bandlimit = bandlimit * 0.5 / np.floor(resolution / 2.0)

    else:
        bandlimit = 0.5
        support_size = resolution // 2

    n_r = int(np.ceil(4 * bandlimit * support_size))
    basis, sample_points = precompute_fb(n_r, support_size, bandlimit)
    _, coeff, mean_coeff, spca_coeff, u, d = jobscript_ffbspca(images, support_size,
                                                                   noise_v_r,
                                                                   basis, sample_points)

    ang_freqs = []
    rad_freqs = []
    vec_d = []
    for i in range(len(d)):
        if len(d[i]) != 0:
            ang_freqs.extend(np.ones(len(d[i]), dtype='int') * i)
            rad_freqs.extend(np.arange(len(d[i])) + 1)
            vec_d.extend(d[i])

    ang_freqs = np.array(ang_freqs)
    rad_freqs = np.array(rad_freqs)
    d = np.array(vec_d)
    k = min(len(d), 400)  # keep the top 400 components
    sorted_indices = np.argsort(-d)
    sorted_indices = sorted_indices[:k]
    d = d[sorted_indices]
    ang_freqs = ang_freqs[sorted_indices]
    rad_freqs = rad_freqs[sorted_indices]

    s_coeff = np.zeros((len(d), num_images), dtype='complex128')

    for i in range(len(d)):
        s_coeff[i] = spca_coeff[ang_freqs[i]][rad_freqs[i] - 1]

    fn = ift_fb(support_size, bandlimit)

    eig_im = np.zeros((np.square(2 * support_size), len(d)), dtype='complex128')

    for i in range(len(d)):
        tmp = fn[ang_freqs[i]]
        tmp = tmp.reshape((int(np.square(2 * support_size)), tmp.shape[2]), order='F')
        eig_im[:, i] = np.dot(tmp, u[ang_freqs[i]][:, rad_freqs[i] - 1])

    fn0 = fn[0].reshape((int(np.square(2 * support_size)), fn[0].shape[2]), order='F')

    spca_data_struct = {'eigval': d, 'freqs': ang_freqs, 'radial_freqs': rad_freqs, 'coeff': s_coeff,
                        'mean': mean_coeff, 'c': bandlimit, 'r': support_size, 'eig_im': eig_im, 'fn0': fn0}
    spca_data = common.create_struct(spca_data_struct)
    return spca_data


def precompute_fb(n_r, support_size, bandlimit):
    sample_points = common.lgwt(n_r, 0, bandlimit)
    basis = bessel_ns_radial(bandlimit, support_size, sample_points.x)
    return basis, sample_points


def bessel_ns_radial(bandlimit, support_size, x):
    bessel = get_bessel()
    bessel = bessel[bessel[:, 3] <= 2 * np.pi * bandlimit * support_size, :]
    angular_freqs = bessel[:, 0]
    max_ang_freq = int(np.max(angular_freqs))
    n_theta = int(np.ceil(16 * bandlimit * support_size))
    if n_theta % 2 == 1:
        n_theta += 1

    radian_freqs = bessel[:, 1]
    r_ns = bessel[:, 2]
    phi_ns = np.zeros((len(x), len(angular_freqs)))
    phi = {}

    angular_freqs_length = len(angular_freqs)
    for i in range(angular_freqs_length):
        r0 = x * r_ns[i] / bandlimit
        f = sp.jv(angular_freqs[i], r0)
        # probably the square and the sqrt not needed
        tmp = np.pi * np.square(sp.jv(angular_freqs[i] + 1, r_ns[i]))
        phi_ns[:, i] = f / (bandlimit * np.sqrt(tmp))

    for i in range(max_ang_freq + 1):
        phi[i] = phi_ns[:, angular_freqs == i]

    struct_dict = {'phi_ns': phi, 'angular_freqs': angular_freqs, 'radian_freqs': radian_freqs, 'n_theta': n_theta}
    return common.create_struct(struct_dict)


def jobscript_ffbspca(images, support_size, noise_var, basis, sample_points, num_threads=10):
    coeff = fbcoeff_nfft(images, support_size, basis, sample_points, num_threads)
    u, d, spca_coeff, mean_coeff = spca_whole(coeff, noise_var)
    return 0, coeff, mean_coeff, spca_coeff, u, d


def fbcoeff_nfft(images, support_size, basis, sample_points, num_threads):
    split_images = np.array_split(images, num_threads, axis=2)
    image_size = split_images[0].shape[0]
    orig = int(np.floor(image_size / 2))
    start_pixel = orig - support_size
    end_pixel = orig + support_size
    new_image_size = int(2 * support_size)

    # unpacking input
    phi_ns = basis.phi_ns
    angular_freqs = basis.angular_freqs
    max_angular_freqs = int(np.max(angular_freqs))
    n_theta = basis.n_theta
    x = sample_points.x
    w = sample_points.w
    w = w * x

    # sampling points in the fourier domain
    freqs = pft_freqs(x, n_theta)
    precomp = common.create_struct({'n_theta': n_theta, 'n_r': len(x), 'resolution': new_image_size, 'freqs': freqs})
    scale = 2 * np.pi / n_theta

    coeff_pos_k = []
    pos_k = []

    for i in range(num_threads):
        curr_images = split_images[i]
        # can work with odd images as well
        curr_images = curr_images[start_pixel:end_pixel, start_pixel:end_pixel, :]
        tmp = cryo_pft_nfft(curr_images, precomp)
        pf_f = scale * np.fft.fft(tmp, axis=1)
        pos_k.append(pf_f[:, :max_angular_freqs + 1, :])

    pos_k = np.concatenate(pos_k, axis=2)

    for i in range(max_angular_freqs + 1):
        coeff_pos_k.append(np.einsum('ki, k, kj -> ij', phi_ns[i], w, pos_k[:, i]))

    return coeff_pos_k


def pft_freqs(x, n_theta):
    n_r = len(x)
    d_theta = 2 * np.pi / n_theta

    # sampling points in the fourier domain
    freqs = np.zeros((n_r * n_theta, 2))
    for i in range(n_theta):
        freqs[i * n_r:(i + 1) * n_r, 0] = x * np.sin(i * d_theta)
        freqs[i * n_r:(i + 1) * n_r, 1] = x * np.cos(i * d_theta)

    return freqs


def cryo_pft_nfft(projections, precomp):
    freqs = precomp.freqs
    n_theta = precomp.n_theta
    n_r = precomp.n_r

    num_projections = projections.shape[2]
    x = -2 * np.pi * freqs.T
    x = x.copy()
    pf = np.empty((x.shape[1], num_projections), dtype='complex128', order='F')
    finufftpy.nufft2d2many(x[0], x[1], pf, -1, 1e-15, projections)
    pf = pf.reshape((n_r, n_theta, num_projections), order='F')

    return pf


def spca_whole(coeff, var_hat):
    max_ang_freq = len(coeff) - 1
    n_p = coeff[0].shape[1]
    mean_coeff = np.mean(coeff[0], axis=1)
    u = []
    d = []
    spca_coeff = []

    for i in range(max_ang_freq + 1):
        tmp = coeff[i]
        lr = tmp.shape[0]
        if i == 0:
            tmp = (tmp.T - mean_coeff).T
            lambda_var = float(lr) / n_p
        else:
            lambda_var = float(lr) / (2 * n_p)

        c1 = np.real(np.einsum('ij, kj -> ik', tmp, np.conj(tmp))) / n_p
        curr_d, curr_u = np.linalg.eig(c1)
        sorted_indices = np.argsort(-curr_d)
        curr_d = curr_d[sorted_indices]
        curr_u = curr_u[:, sorted_indices]

        if var_hat != 0:
            k = np.count_nonzero(curr_d > var_hat * np.square(1 + np.sqrt(lambda_var)))
            if k != 0:
                curr_d = curr_d[:k]
                curr_u = curr_u[:, :k]
                d.append(curr_d)
                u.append(curr_u)
                l_k = 0.5 * ((curr_d - (lambda_var + 1) * var_hat) + np.sqrt(
                    np.square((lambda_var + 1) * var_hat - curr_d) - 4 * lambda_var * np.square(var_hat)))
                snr_i = l_k / var_hat
                snr = (np.square(snr_i) - lambda_var) / (snr_i + lambda_var)
                weight = 1 / (1 + 1 / snr)
                spca_coeff.append(np.einsum('i, ji, jk -> ik', weight, curr_u, tmp))
            else:
                d.append([])
                u.append([])
                spca_coeff.append([])
        else:
            u.append(curr_u)
            d.append(curr_d)
            spca_coeff.append(np.einsum('ji, jk -> ik', curr_u, tmp))

    return u, d, spca_coeff, mean_coeff


def ift_fb(support_size, bandlimit):
    support_size = int(support_size)
    x, y = np.meshgrid(np.arange(-support_size, support_size), np.arange(-support_size, support_size))
    r = np.sqrt(np.square(x) + np.square(y))
    inside_circle = r <= support_size
    theta = np.arctan2(x, y)
    theta = theta[inside_circle]
    r = r[inside_circle]

    bessel = get_bessel()
    bessel = bessel[bessel[:, 3] <= 2 * np.pi * bandlimit * support_size, :]
    k_max = int(np.max(bessel[:, 0]))
    fn = []

    computation1 = 2 * np.pi * bandlimit * r
    computation2 = np.square(computation1)
    c_sqrt_pi_2 = 2 * bandlimit * np.sqrt(np.pi)
    bessel_freqs = bessel[:, 0]
    bessel2 = bessel[:, 2]
    for i in range(k_max + 1):
        bessel_k = bessel2[bessel_freqs == i]
        tmp = np.zeros((2 * support_size, 2 * support_size, len(bessel_k)), dtype='complex128')

        f_r_base = c_sqrt_pi_2 * np.power(-1j, i) * sp.jv(i, computation1)
        f_theta = np.exp(1j * i * theta)

        tmp[inside_circle, :] = np.outer(f_r_base * f_theta, np.power(-1, np.arange(1, len(bessel_k) + 1)) * bessel_k) \
                                / np.subtract.outer(computation2, np.square(bessel_k))

        fn.append(np.transpose(tmp, axes=(1, 0, 2)))

    return fn


def get_bessel():
    try:
        return np.load(BESSEL_NPY)
    except:
        raise OSError('Please run only from aspire_refactored!')