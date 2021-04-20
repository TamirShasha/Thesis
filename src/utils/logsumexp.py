import numpy as np
import numba as nb


@nb.jit
def logsumexp(a):
    """
    To reduce running time I implemented logsumexp myself, the scipy version has too much additional things I don't need
    :param a:
    :param axis:
    :param keepdims:
    :return:
    """
    a_max = np.max(a)
    if np.isneginf(a_max):
        return -np.inf
    output = np.log(np.sum(np.exp(a - a_max)))
    output += a_max
    return output
