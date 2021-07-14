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


# utils for optimization
def precompute_f_f_tag(power, mgraph, filt, noise_std):
    x_tag = np.flip(filt)  # Flipping to cross-correlate
    if len(x_tag.shape) == 1 and len(mgraph.shape) == 2:
        conv = np.array([convolve(mgraph[i], x_tag, mode='valid') for i in range(mgraph.shape[0])])
    else:
        conv = convolve(mgraph, x_tag, mode='valid')
    conv /= noise_std ** 2
    return conv, conv * power


@nb.jit
def dynamic_programming_1d_derivative(n, k, d, consts1, consts2, g=None):
    """
    Compute log(\sum_{s in S_(n,k,d)}\prod_{i in s}c_i)
    :param n:
    :param k:
    :param d:
    :param consts1:
    :param consts2:
    :param g:
    :return:
    """
    g = dynamic_programming_1d(n, k, d, consts2) if g is None else g

    r = - np.min(consts1) + 1
    consts1 = np.log(consts1 + r)

    mapping = np.full((n + 1, k + 1), -np.inf)
    mapping[:, 0] = 0

    for curr_k in range(1, k + 1):
        for i in range(n - curr_k * d, -1, -1):
            mapping[i, curr_k] = np.logaddexp(consts2[i] + mapping[i + d, curr_k - 1], mapping[i + 1, curr_k])
            mapping[i, curr_k] = np.logaddexp(consts1[i] + consts2[i] + g[i + d, curr_k - 1], mapping[i, curr_k])

    return np.exp(mapping[0, k] - g[0, k]) - k * r


# @nb.jit
def dynamic_programming_1d_many_derivative_for_2d_derivative(n, k, d, consts1, consts2, g=None):
    """
    Do the 1d dynamic programming for many constant vectors.
    Output is mapping such that mapping[:, :, i] = dynamic_programming_1d(n, k, d, constants[i])
    :param n:
    :param k:
    :param d:
    :param constants:
    :return:
    """

    g = dynamic_programming_1d_many(n, k, d, consts2) if g is None else g
    g = g.transpose((1, 2, 0)).copy()

    # Changing constants shape so the first axis is continuous
    consts1 = consts1.T.copy()
    consts2 = consts2.T.copy()

    # Allocating memory
    # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
    # when k=0 the probability is 1
    mapping = np.full((n + 1, k + 1, consts1.shape[-1]), -np.inf)
    mapping[:, 0] = 0

    for curr_k in range(1, k + 1):
        for i in range(n - curr_k * d, -1, -1):
            mapping[i, curr_k] = np.logaddexp(consts2[i] + mapping[i + d, curr_k - 1], mapping[i + 1, curr_k])
            mapping[i, curr_k] = np.logaddexp(consts1[i] + consts2[i] + g[i + d, curr_k - 1], mapping[i, curr_k])

    mapping = mapping.transpose((2, 0, 1))
    return mapping[:, 0, :].copy()


def dynamic_programming_2d_function_and_derivative(n, k, d, consts1, consts2):
    max_k_in_row = min(n // d, k)

    # Fix consts1
    r = - np.min(consts1) + 1
    consts1 = np.log(consts1 + r)

    # Start precomputation
    C = dynamic_programming_1d_many(n, k, d, consts2)
    A = dynamic_programming_1d_many_derivative_for_2d_derivative(n, k, d, consts1, consts2, C)
    # C = C[:, 0, 1:].copy()  # No need for k = 0
    # B = dynamic_programming_2d_after_pre_compute(n, k, d, C)
    B = dynamic_programming_2d_after_pre_compute(n, k, d, C[:, 0, 1:].copy())

    # A = A[:, :0:-1].copy()  # No need for k = 0 and need to reverse columns
    C = C[:, 0, :].copy()

    # start DP
    mapping = np.full((n + 1, k + 1), -np.inf)
    mapping[:, 0] = 0

    # Filling values one by one, skipping irrelevant values
    for curr_k in range(1, k + 1):
        max_k = min(curr_k, max_k_in_row)
        tmp = np.zeros(max_k + 1)
        for i in range(n - d, -1, -1):
            tmp[0] = mapping[i + 1, curr_k]
            for k_ in range(1, max_k + 1):
                tmp[k_] = np.logaddexp(A[i, k_] + B[i + d, curr_k - k_], C[i, k_] + mapping[i + d, curr_k - k_])
            mapping[i, curr_k] = logsumexp_simple(tmp)
            # mapping[i, curr_k] = logsumexp_simple(
            #     np.logaddexp(A[i, curr_k - max_k:curr_k] + B[i + d, -max_k:],
            #                  mapping[i + d, curr_k - max_k:curr_k] + C[i, -max_k:])
            # )
            # mapping[i, curr_k] = np.logaddexp(mapping[i, curr_k], mapping[i + 1, curr_k])
            # print(logsumexp_simple(tmp) - mapping[i, curr_k])

    return B[0, k], np.exp(mapping[0, k] - B[0, k]) - k * r


def gradient_descent(F_F_tag, initial_x, t=0.1, epsilon=1e-10, max_iter=200, concave=False):
    x_prev = initial_x
    F_prev, F_tag_prev = F_F_tag(x_prev)
    for i in range(max_iter):
        print(x_prev, F_prev, F_tag_prev, t)
        x_current = x_prev + t * F_tag_prev if concave else x_prev - t * F_tag_prev
        F_current, F_tag_current = F_F_tag(x_current)
        if np.abs(F_current - F_prev) < epsilon:
            break
        t = np.abs((x_current - x_prev) / (F_tag_prev - F_tag_current))
        x_prev, F_prev, F_tag_prev = x_current, F_current, F_tag_current
    print(x_current, F_current, F_tag_current, t)
    return F_current, x_current


# Code for 1d optimization
def max_argmax_1d_case(y, filt, k, noise_std, x_0=0, t=0.1, epsilon=1e-5, max_iter=100):
    # If got only one y
    if not hasattr(y[0], '__iter__'):
        y = [y]

    # If got only one k
    if not hasattr(k, '__iter__'):
        k = [k] * len(y)

    # Computing log(|S|)
    d = filt.shape[0]
    num_curves = len(y)
    log_sizes = np.zeros(num_curves)
    for i in range(num_curves):
        n = len(y[i])
        log_sizes[i] = log_size_S_1d(n, k[i], d)

    def F_F_tag(x):
        return f_f_tag_1d(x, y[i], filt, k[i], noise_std, log_sizes)

    return gradient_descent(F_F_tag, x_0, t, epsilon, max_iter, concave=True)


def f_f_tag_1d_one_sample(curr_power, y, x, k, sigma, log_size=0):
    n = y.shape[0]
    d = x.shape[0]

    # Computing constant part
    log_c = - np.sum(np.square(y)) / (2 * sigma ** 2) - np.log(np.sqrt(2 * np.pi * sigma ** 2))

    const1, const2 = precompute_f_f_tag(curr_power, y, x, sigma)
    g = dynamic_programming_1d(n, k, d, const2)
    dp_derivative = dynamic_programming_1d_derivative(n, k, d, const1, const2, g)

    log_f = - log_size + log_c - k * np.sum(np.square(x)) * curr_power ** 2 / (2 * sigma ** 2) + g[0, k]
    f_tag = - k * np.sum(np.square(x)) * curr_power / sigma ** 2 + dp_derivative

    return log_f, f_tag


def f_f_tag_1d(curr_power, y, x, k, sigma, log_sizes=None):
    num_curves = len(y)
    if log_sizes is None:
        d = x.shape[0]
        log_sizes = np.zeros(num_curves)
        for i in range(num_curves):
            n = len(y[i])
            log_sizes[i] = log_size_S_1d(n, k[i], d)

    f = np.array(num_curves)
    f_tag = np.array(num_curves)
    for i in range(num_curves):
        f_, f_tag_ = f_f_tag_1d(curr_power, y[i], x, k[i], sigma, log_sizes[i])
        f[i] += f_ / num_curves
        f_tag[i] += f_tag_ / num_curves
    return f.sum(), f_tag.sum()


# Code for 2d optimization
def max_argmax_2d_case(y, filt, k, noise_std, x_0=0, t=0.1, epsilon=1e-5, max_iter=100):
    n = y.shape[0]
    d = filt.shape[0]
    log_size_1_axis = log_size_S_2d_1axis(n, k, d)

    def F_F_tag(x):
        return f_f_tag_2d(x, y, filt, k, noise_std, log_size_1_axis)

    return gradient_descent(F_F_tag, x_0, t, epsilon, max_iter, concave=True)


def f_f_tag_2d(curr_power, y, x, k, sigma, log_size_1_axis=None):
    n = y.shape[0]
    d = x.shape[0]
    if log_size_1_axis is None:
        log_size_1_axis = log_size_S_2d_1axis(n, k, d)

    # Computing constant part
    log_c = - np.sum(np.square(y)) / (2 * sigma ** 2) - np.log(np.sqrt(2 * np.pi * sigma ** 2))

    # Axis 1
    const1, const2 = precompute_f_f_tag(curr_power, y, x, sigma)
    log_f1, f_tag1 = dynamic_programming_2d_function_and_derivative(n, k, d, const1, const2)
    log_f1 += - log_size_1_axis + log_c - k * np.sum(np.square(x)) * curr_power ** 2 / (2 * sigma ** 2)
    f_tag1 += - k * np.sum(np.square(x)) * curr_power / sigma ** 2

    # Axis 2
    const1, const2 = const1.T.copy(), const2.T.copy()
    log_f2, f_tag2 = dynamic_programming_2d_function_and_derivative(n, k, d, const1, const2)
    log_f2 += - log_size_1_axis + log_c - k * np.sum(np.square(x)) * curr_power ** 2 / (2 * sigma ** 2)
    f_tag2 += - k * np.sum(np.square(x)) * curr_power / sigma ** 2

    log_f = log_f1 + log_f2
    f_tag = f_tag1 + f_tag2
    return log_f, f_tag
