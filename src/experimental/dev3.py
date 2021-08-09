import numpy as np


def heuristic_dp(n, d, k, c):
    """

    :param n:
    :param d:
    :param k:
    :param c: ndarray of size (n-d+1, T), c[i, t] = convolution between filter t and the i'th part of the mgraph.
    :return:
    """
    T = c.shape[1]
    g = np.zeros((n + 1, k + 1))
    b = np.zeros((n + 1, k + 1, T))
    norm_c = np.sum(np.square(c), 1)

    g[:, 0] = 1

    indices = []
    for l in range(k + 1):
        indices.append(np.zeros((n + 1, l), np.integer))

    # Filling values one by one, skipping irrelevant values
    # We already filled values when k=0 (=0) and when i>n-k*d
    for curr_k in range(1, k + 1):
        for i in range(n - curr_k * d, -1, -1):
            val1 = g[i + 1, curr_k]
            val2 = norm_c[i] + np.inner(c[i], b[i + d, curr_k - 1]) + g[i + d, curr_k - 1]
            if val1 > val2:
                g[i, curr_k] = val1
                b[i, curr_k] = b[i + 1, curr_k]
                indices[curr_k][i] = indices[curr_k][i + 1]
            else:
                g[i, curr_k] = val2
                b[i, curr_k] = b[i + d, curr_k - 1] + c[i]
                indices[curr_k][i, 1:] = indices[curr_k-1][i + d]
                indices[curr_k][i, 0] = i
    return g[0, k], indices[k][0]


def heuristic2_dp(n, d, k, c):
    """

    :param n:
    :param d:
    :param k:
    :param c: ndarray of size (n-d+1, T), c[i, t] = convolution between filter t and the i'th part of the mgraph.
    :return:
    """
    g = np.zeros((n + 1, k + 1))
    norm_c = np.sum(np.square(c), 1)

    g[:, 0] = 1

    indices = []
    for l in range(k + 1):
        indices.append(np.zeros((n + 1, l), np.integer))

    # Filling values one by one, skipping irrelevant values
    # We already filled values when k=0 (=0) and when i>n-k*d
    for curr_k in range(1, k + 1):
        for i in range(n - curr_k * d, -1, -1):
            val1 = g[i + 1, curr_k]
            val2 = norm_c[i] + g[i + d, curr_k - 1]
            if val1 > val2:
                g[i, curr_k] = val1
                indices[curr_k][i] = indices[curr_k][i + 1]
            else:
                g[i, curr_k] = val2
                indices[curr_k][i, 1:] = indices[curr_k-1][i + d]
                indices[curr_k][i, 0] = i
    return g[0, k], indices[k][0]
