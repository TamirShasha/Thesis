import numpy as np
from scipy.signal import convolve
import numba as nb
from src.utils.logsumexp import logsumexp_simple


def relative_error(estimated_signal, true_signal):
    """
    Calculate the relative error between estimated signal and true signal up to circular shift
    :return: relative error
    """
    n = len(true_signal)
    corr = [np.linalg.norm(true_signal - np.roll(estimated_signal, i)) for i in range(n)]
    shift = np.argmin(corr)
    error = np.min(corr) / np.linalg.norm(true_signal)
    return error, shift


# utils, private, change names
def generate_random_bars(n, k):
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


# utils, change names
def random_1d_ws_positions(n, k, d):
    signal_mask = generate_random_bars(n - d * k, k + 1)
    s_cum = np.cumsum(signal_mask)
    positions = np.zeros(k)
    for i in np.arange(s_cum.shape[0] - 1):
        start = s_cum[i] + d * i
        positions[i] = start
    return positions


# only first function uses, private, utils
def log_num_k_sums_to_n(n, k):
    """
    Compute the log number of #{k tuples that sum to n}.
    """
    n_tag = n + k - 1
    k_tag = k - 1
    return log_binomial(n_tag, k_tag)


# private, utils
def log_binomial(n, k):
    """
    Compute the log of the binomial coefficient.
    """
    nominator = np.sum(np.log(np.arange(n) + 1))
    denominator = np.sum(np.log(np.arange(k) + 1)) + np.sum(np.log(np.arange(n - k) + 1))
    return nominator - denominator


# utils
def downample_signal(signal, d):
    """
    Take arbitrary signal of any length and change it to length d. For now only support downsampling it and not
    upsampling.
    """
    return cryo_downsample(signal, (d,))


# utils
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


# utils
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


# math
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


# utils for 2d vws, private, names
def log_size_S_2d_1axis(n, k, d):
    """
    Compute log(|S|), where |S| is the number of ways to insert k signals of size (d x d) in (n x n) spaces in such they are
    very well separated on rows.
    """
    if k * d ** 2 > n ** 2:
        return -np.inf
    max_k_in_row = min(n // d, k)
    log_size_S_per_row_per_k = np.zeros((n - d + 1, max_k_in_row))
    for k_in_row in range(1, max_k_in_row + 1):
        log_size_S_per_row_per_k[:, k_in_row - 1] = log_size_S_1d(n, k_in_row, d)

    mapping = _calc_mapping_2d_after_precompute(n, k, d, log_size_S_per_row_per_k)
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


# private, names
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


# names,
# @nb.jit
# def calc_mapping_1d(n, k, d, constants):
#     """
#     Compute log(\sum_{s in S_(n,k,d)}\prod_{i in s}c_i)
#     :param n:
#     :param k:
#     :param d:
#     :param constants:
#     :return:
#     """
#     # Allocating memory
#     # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
#     # when k=0 the probability is 1
#     mapping = np.full((n + 1, k + 1), -np.inf)
#     mapping[:, 0] = 0
#
#     # Filling values one by one, skipping irrelevant values
#     # We already filled values when k=0 (=0) and when i>n-k*d
#     for curr_k in range(1, k + 1):
#         for i in range(n - curr_k * d, -1, -1):
#             mapping[i, curr_k] = np.logaddexp(constants[i] + mapping[i + d, curr_k - 1], mapping[i + 1, curr_k])
#
#     return mapping

@nb.jit
def calc_mapping_1d(n, k, d, constants):
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
            # mapping[i, curr_k] = np.maximum(constants[i] + mapping[i + d, curr_k - 1], mapping[i + 1, curr_k])

    return mapping


@nb.jit
def _calc_mapping_1d_many(n, k, d, constants):
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


def calc_mapping_2d(n, k, d, row_constants):
    """
    creates the mapping of size (n+1 x k+1)
    first coordinate refers to the row, second refers to amount of signals left to insert
    """
    max_k_in_row = min(n // d, k)
    mapping_per_row = _calc_mapping_1d_many(n, max_k_in_row, d, row_constants)[:, 0, 1:]

    return _calc_mapping_2d_after_precompute(n, k, d, mapping_per_row)


# @nb.jit
def _calc_mapping_2d_after_precompute(n, k, d, constants):
    """
    computes 2d mapping with precomputed constants
    """
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


# Utils for optimization
def _calc_constants(data, signal_filter, filter_coeff, noise_std):
    flipped_signal_filter = np.flip(signal_filter)  # Flipping to cross-correlate
    if len(flipped_signal_filter.shape) == 1 and len(data.shape) == 2:
        conv = np.array([convolve(data[i], flipped_signal_filter, mode='valid') for i in range(data.shape[0])])
    else:
        conv = convolve(data, flipped_signal_filter, mode='valid')

    prod_consts = (-2 * conv * filter_coeff + np.sum(np.square(flipped_signal_filter)) * filter_coeff ** 2) / (
            -2 * noise_std ** 2)
    sum_consts = (-2 * conv + 2 * np.sum(np.square(flipped_signal_filter)) * filter_coeff) / (-2 * noise_std ** 2)
    return sum_consts, prod_consts


@nb.jit
def _calc_term_two_derivative_1d(n, k, d, sum_consts, prod_consts, mapping=None):
    """
    Compute log(\sum_{s in S_(n,k,d)}\prod_{i in s}c_i)
    :param n:
    :param k:
    :param d:
    :param sum_consts:
    :param prod_consts:
    :param mapping:
    :return:
    """
    mapping = calc_mapping_1d(n, k, d, prod_consts) if mapping is None else mapping

    r = - np.min(sum_consts) + 1
    log_sum_consts = np.log(sum_consts + r)

    derivative_mapping = np.full((n + 1, k + 1), -np.inf)
    derivative_mapping[:, 0] = 0

    for curr_k in range(1, k + 1):
        for i in range(n - curr_k * d, -1, -1):
            derivative_mapping[i, curr_k] = np.logaddexp(prod_consts[i] + derivative_mapping[i + d, curr_k - 1],
                                                         derivative_mapping[i + 1, curr_k])
            derivative_mapping[i, curr_k] = np.logaddexp(
                log_sum_consts[i] + prod_consts[i] + mapping[i + d, curr_k - 1],
                derivative_mapping[i, curr_k])

    term2 = np.exp(derivative_mapping[0, k] - mapping[0, k]) - k * r
    return term2


# @nb.jit
def _calc_likelihood_and_likelihood_derivative_without_constants_2d_pre_compute(n, k, d, prod_consts, sum_consts,
                                                                                g=None):
    """
    Do the 1d dynamic programming for many constant vectors.
    Output is mapping such that mapping[:, :, i] = dynamic_programming_1d(n, k, d, constants[i])
    :param n:
    :param k:
    :param d:
    :param constants:
    :return:
    """

    g = _calc_mapping_1d_many(n, k, d, sum_consts) if g is None else g
    g = g.transpose((1, 2, 0)).copy()

    # Changing constants shape so the first axis is continuous
    prod_consts = prod_consts.T.copy()
    sum_consts = sum_consts.T.copy()

    # Allocating memory
    # Default is -inf everywhere as there are many places where the probability is 0 (when i > n - k * d)
    # when k=0 the probability is 1
    mapping = np.full((n + 1, k + 1, prod_consts.shape[-1]), -np.inf)
    mapping[:, 0] = 0

    for curr_k in range(1, k + 1):
        for i in range(n - curr_k * d, -1, -1):
            mapping[i, curr_k] = np.logaddexp(sum_consts[i] + mapping[i + d, curr_k - 1], mapping[i + 1, curr_k])
            mapping[i, curr_k] = np.logaddexp(prod_consts[i] + sum_consts[i] + g[i + d, curr_k - 1], mapping[i, curr_k])

    mapping = mapping.transpose((2, 0, 1))
    return mapping[:, 0, :].copy()


def _calc_likelihood_and_likelihood_derivative_without_constants_2d(n, k, d, prod_consts, sum_consts):
    max_k_in_row = min(n // d, k)

    # Fix prod_consts
    r = - np.min(prod_consts) + 1
    prod_consts = np.log(prod_consts + r)

    # Start precomputation
    C = _calc_mapping_1d_many(n, k, d, sum_consts)
    A = _calc_likelihood_and_likelihood_derivative_without_constants_2d_pre_compute(n, k, d, prod_consts, sum_consts, C)
    # C = C[:, 0, 1:].copy()  # No need for k = 0
    # B = dynamic_programming_2d_after_pre_compute(n, k, d, C)
    B = _calc_mapping_2d_after_precompute(n, k, d, C[:, 0, 1:].copy())

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


def _gradient_descent(F_F_derivative, initial_x, t=0.1, epsilon=1e-10, max_iter=200, concave=False):
    """
    :param F_F_derivative: returns function value and its derivative at a point
    :param initial_x: initial guess
    :param t: step size
    :param epsilon: tolerance
    :param max_iter: maximum iterations to run
    :param concave: TODO:
    :return:
    """
    x_prev = initial_x
    F_prev, F_tag_prev = F_F_derivative(x_prev)
    for i in range(max_iter):
        # print(x_prev, F_prev, F_tag_prev, t)
        x_current = x_prev + t * F_tag_prev if concave else x_prev - t * F_tag_prev
        F_current, F_tag_current = F_F_derivative(x_current)
        # print(f'at iteration # {i + 1}, {np.abs(F_current - F_prev)}')
        if np.abs(F_current - F_prev) < epsilon:
            break
        t = np.abs(np.linalg.norm(x_current - x_prev) / np.linalg.norm(F_tag_prev - F_tag_current))
        x_prev, F_prev, F_tag_prev = x_current, F_current, F_tag_current
    # print(x_current, F_current, F_tag_current, t)
    return F_current, x_current


# Code for 1d optimization
def calc_most_likelihood_and_optimized_power_1d(data, signal_filter, k, noise_std, p_0=0, t=0.1, epsilon=1e-5,
                                                max_iter=100):
    """
    :param data: Given micrograph
    :param signal_filter: base signal filter
    :param k: num of occurrences, can be fixed int or array of integers
    :param noise_std:
    :param p_0: initial power guess
    :param t: first step size
    :param epsilon: tolerance
    :param max_iter: max iterations to run
    :return:
    """
    # If got only one y
    if not hasattr(data[0], '__iter__'):
        data = [data]

    # If got only one k
    if not hasattr(k, '__iter__'):
        k = [k] * len(data)

    def F_F_derivative(x):
        return _calc_f_f_derivative_1d(x, data, signal_filter, k, noise_std)

    f_opt, p_opt = _gradient_descent(F_F_derivative, p_0, t, epsilon, max_iter, concave=True)

    # Computing constant part
    d = signal_filter.shape[0]
    num_curves = len(data)
    log_sizes = np.zeros(num_curves)
    for i in range(num_curves):
        n = len(data[i])
        log_sizes[i] = log_size_S_1d(n, k[i], d)

    constant_part = (log_prob_all_is_noise(data, noise_std) - np.sum(log_sizes)) / num_curves
    f_opt += constant_part
    return f_opt, p_opt


def _calc_f_f_derivative_1d_one_sample(curr_power, data, signal_filter, k, noise_std):
    n = data.shape[0]
    d = signal_filter.shape[0]

    sum_consts, prod_consts = _calc_constants(data, signal_filter, curr_power, noise_std)
    mapping = calc_mapping_1d(n, k, d, prod_consts)
    f_tag = _calc_term_two_derivative_1d(n, k, d, sum_consts, prod_consts, mapping)
    log_f = mapping[0, k]

    return log_f, f_tag


def _calc_f_f_derivative_1d(curr_power, data, signal_filter, k, noise_std):
    num_curves = len(data)

    f = np.zeros(num_curves)
    f_tag = np.zeros(num_curves)
    for i in range(num_curves):
        f[i], f_tag[i] = _calc_f_f_derivative_1d_one_sample(curr_power, data[i], signal_filter, k[i], noise_std)

    return f.mean(), f_tag.mean()


# Code for 2d optimization
def calc_most_likelihood_and_optimized_power_2d(data, signal_filter, k, noise_std, p_0=0, t=0.1, epsilon=1e-5,
                                                max_iter=100):
    """
    :param data: Given micrograph
    :param signal_filter: base signal filter
    :param k: num of occurrences, can be fixed int or array of integers
    :param noise_std:
    :param p_0: initial power guess
    :param t: first step size
    :param epsilon: tolerance
    :param max_iter: max iterations to run
    """

    def F_F_derivative(x):
        return _calc_f_f_derivative_2d(x, data, signal_filter, k, noise_std)

    f_opt, p_opt = _gradient_descent(F_F_derivative, p_0, t, epsilon, max_iter, concave=True)

    # Compute constant part
    log_size_1_axis = log_size_S_2d_1axis(data.shape[0], k, signal_filter.shape[0])
    constant_part = log_prob_all_is_noise(data, noise_std) - (log_size_1_axis + np.log(2))

    f_opt += constant_part
    return f_opt, p_opt


def _calc_f_f_derivative_2d(curr_power, data, signal_filter, k, noise_std):
    n = data.shape[0]
    d = signal_filter.shape[0]

    # Axis 1
    sum_consts, prod_consts = _calc_constants(data, signal_filter, curr_power, noise_std)
    log_f_rows, f_derivative_rows = _calc_likelihood_and_likelihood_derivative_without_constants_2d(n, k, d, sum_consts,
                                                                                                    prod_consts)

    # Axis 2
    sum_consts, prod_consts = sum_consts.T.copy(), prod_consts.T.copy()
    log_f_columns, f_derivative_columns = _calc_likelihood_and_likelihood_derivative_without_constants_2d(n, k, d,
                                                                                                          sum_consts,
                                                                                                          prod_consts)

    # Combining the axes
    r = - np.min(sum_consts) + 1
    tmp1 = np.log(f_derivative_rows + k * r) + log_f_rows
    tmp2 = np.log(f_derivative_columns + k * r) + log_f_columns

    log_f = np.logaddexp(log_f_rows, log_f_columns)
    f_derivative = np.exp(np.logaddexp(tmp1, tmp2) - log_f) - k * r
    return log_f, f_derivative
