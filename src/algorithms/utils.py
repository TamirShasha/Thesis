import numpy as np
from scipy.signal import convolve
import numba as nb
from src.utils.logsumexp import logsumexp_simple


def create_random_k_tuple_sum_to_n(n, k):
    """
    Output a random k tuple of non-negative integers that sums to n, with uniform probability over all options.
    """
    output = []
    while True:
        if k == 1:
            output.append(n)
            break

        log_num_options_total = log_num_k_sums_to_n(n, k)

        log_num_options_x0_is_i = np.zeros(n + 1)
        # Saving computations with the relation: log_num_options_x0_is_i[0] = log_num_options_x0_is_i * (k-1) / (n+1)
        log_num_options_x0_is_i[0] = log_num_options_total + np.log(k - 1) - np.log(n + 1)
        for i in range(n):
            # Using the relation: log_num_options_x0_is_i[i+1] = log_num_options_x0_is_i[i] * (n-i+1) / (n-i+k-1)
            log_num_options_x0_is_i[i + 1] = log_num_options_x0_is_i[i] + np.log(n - i + 1) - np.log(n - i + k - 1)

        prob_x0_is_i = np.exp(log_num_options_x0_is_i - log_num_options_total)
        # There is some numerical problems with this fast computation, so simply normalize
        prob_x0_is_i /= prob_x0_is_i.sum()
        head = np.random.choice(np.arange(n + 1), p=prob_x0_is_i)
        output.append(head)

        # Updating n and k
        k -= 1
        n -= head
    return np.array(output)


def random_1d_ws_positions(n, k, d):
    signal_mask = create_random_k_tuple_sum_to_n(n - d * k, k + 1)
    s_cum = np.cumsum(signal_mask)
    positions = np.zeros(k)
    for i in np.arange(s_cum.shape[0] - 1):
        start = s_cum[i] + d * i
        positions[i] = start
    return positions


def log_num_k_sums_to_n(n, k):
    """
    Compute the log number of #{k tuples that sum to n}.
    """
    n_tag = n + k - 1
    k_tag = k - 1
    return log_binomial(n_tag, k_tag)


def log_binomial(n, k):
    """
    Compute the log of the binomial coefficient.
    """
    nominator = np.sum(np.log(np.arange(n) + 1))
    denominator = np.sum(np.log(np.arange(k) + 1)) + np.sum(np.log(np.arange(n - k) + 1))
    return nominator - denominator


def downample_signal(signal, d):
    """
    Take arbitrary signal of any length and change it to length d. For now only support downsampling it and not
    upsampling.
    """
    return cryo_downsample(signal, (d,))


def cryo_downsample(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i (n_i) is the size we want to cut from
        the center of x in dimension i. If the value of n_i <= 0 or >= N_i then the dimension is left as is.
    :return: out: downsampled x
    """
    dtype_in = x.dtype
    in_shape = np.array(x.shape)
    out_shape = np.array([s if 0 < s < in_shape[i] else in_shape[i] for i, s in enumerate(out_shape)])
    fourier_dims = np.array([i for i, s in enumerate(out_shape) if 0 < s < in_shape[i]])
    size_in = np.prod(in_shape[fourier_dims])
    size_out = np.prod(out_shape[fourier_dims])

    fx = crop(np.fft.fftshift(np.fft.fft2(x, axes=fourier_dims), axes=fourier_dims), out_shape)
    out = np.fft.ifft2(np.fft.ifftshift(fx, axes=fourier_dims), axes=fourier_dims) * (size_out / size_in)
    return out.astype(dtype_in)


def crop(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i is the size we want to cut from the
        center of x in dimension i. If the value is <= 0 then the dimension is left as is
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if s > 0 else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out


# Utils regarding likelihood computation
def log_size_S_1d(n, k, d):
        """
        Compute log(|S|), where |S| is the number of ways to insert k signals of length d in n spaces in such they are
        not overlapping.
        """
        if k * d > n:
            return -np.inf
        n_tag = n - (d - 1) * k
        k_tag = k
        return log_binomial(n_tag, k_tag)


def log_size_S_2d_1axis(n, k, d):
    if k * d ** 2 > n ** 2:
        return -np.inf
    max_k_in_row = min(n // d, k)
    log_size_S_per_row_per_k = np.zeros((n - d + 1, max_k_in_row))
    for k_in_row in range(1, max_k_in_row + 1):
        log_size_S_per_row_per_k[:, k_in_row - 1] = log_size_S_1d(n, k_in_row, d)

    mapping = dynamic_programming_2d_after_pre_compute(n, k, d, log_size_S_per_row_per_k)
    return mapping[0, k]


def log_probability_filter_on_each_pixel(mgraph, filt, noise_std):
    """
    For each pixel i in the mgraph compute -1/2std^2 * \sum_{j is filter pixel} x_j^2 - 2x_jy_{i+j}
    We ignore pixels that cannot start a filter.
    :param mgraph:
    :param filt:
    :param noise_std:
    :return:
    """
    x_tag = np.flip(filt)  # Flipping to cross-correlate
    sum_yx_minus_x_squared = convolve(-2 * mgraph, x_tag, mode='valid')  # -2y to save computations later
    sum_yx_minus_x_squared += np.sum(np.square(filt))
    sum_yx_minus_x_squared *= - 0.5 / noise_std ** 2
    return sum_yx_minus_x_squared


def log_prob_all_is_noise(y, noise_std):
    """
    Compute log(\prod_{i} prob[y_i~Gaussian(0, noise_std)])
    :param y:
    :param noise_std:
    :return:
    """
    if noise_std == 0:
        if np.max(np.abs(y)) == 0:
            return 0
        return -np.inf

    n = y.size
    minus_1_over_twice_variance = - 0.5 / noise_std ** 2
    return - n * np.log(noise_std * (2 * np.pi) ** 0.5) + minus_1_over_twice_variance * np.sum(np.square(y))


@nb.jit
def dynamic_programming_1d(n, k, d, constants):
    """
    Compute log(\sum_{s in S_(n,k,d)}\prod_{i in s}c_i)
    :param n:
    :param k:
    :param d:
    :param constants:
    :return:
    """
    # Allocating memory
    # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
    # when k=0 the probability is 1
    mapping = np.full((n + 1, k + 1), -np.inf)
    mapping[:, 0] = 0

    # Filling values one by one, skipping irrelevant values
    # We already filled values when k=0 (=0) and when i>n-k*d
    for curr_k in range(1, k + 1):
        for i in range(n - curr_k * d, -1, -1):
            mapping[i, curr_k] = np.logaddexp(constants[i] + mapping[i + d, curr_k - 1], mapping[i + 1, curr_k])

    return mapping


@nb.jit
def dynamic_programming_1d_many(n, k, d, constants):
    """
    Do the 1d dynamic programming for many constant vectors.
    Output is mapping such that mapping[:, :, i] = dynamic_programming_1d(n, k, d, constants[i])
    :param n:
    :param k:
    :param d:
    :param constants:
    :return:
    """
    # Changing constants shape so the first axis is continuous
    constants = constants.T.copy()
    # Allocating memory
    # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
    # when k=0 the probability is 1
    mapping = np.full((n + 1, k + 1, constants.shape[-1]), -np.inf)
    mapping[:, 0] = 0

    # Filling values one by one, skipping irrelevant values
    # We already filled values when k=0 (=0) and when i>n-k*d
    for curr_k in range(1, k + 1):
        for i in range(n - curr_k * d, -1, -1):
            mapping[i, curr_k] = np.logaddexp(constants[i] + mapping[i + d, curr_k - 1], mapping[i + 1, curr_k])

    mapping = mapping.transpose((2, 0, 1)).copy()
    return mapping


def dynamic_programming_2d(n, k, d, constants):
    max_k_in_row = min(n // d, k)
    pre_compute_per_row_per_k = dynamic_programming_1d_many(n, max_k_in_row, d, constants)
    pre_compute_per_row_per_k = pre_compute_per_row_per_k[:, 0, 1:]

    return dynamic_programming_2d_after_pre_compute(n, k, d, pre_compute_per_row_per_k)


# @nb.jit
def dynamic_programming_2d_after_pre_compute(n, k, d, constants):
    max_k_in_row = min(n // d, k)
    constants = constants[:, ::-1].copy()
    # Allocating memory
    # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - d)
    # when k=0 the probability is 1
    mapping = np.full((n + 1, k + 1), -np.inf)
    mapping[:, 0] = 0

    # Filling values one by one, skipping irrelevant values
    for curr_k in range(1, k + 1):
        max_k = min(curr_k, max_k_in_row)
        for i in range(n - d, -1, -1):
            mapping[i, curr_k] = logsumexp_simple(mapping[i + d, curr_k - max_k:curr_k] + constants[i, -max_k:])
            mapping[i, curr_k] = np.logaddexp(mapping[i, curr_k], mapping[i + 1, curr_k])

    return mapping
