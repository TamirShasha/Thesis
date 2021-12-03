from skimage.draw import disk
import numpy as np

from src.algorithms.utils import gram_schmidt


def create_filter_basis(filter_length, basis_size, basis_type='chebyshev'):
    """
    creates basis for symmetric signals of given size
    the higher the size, the finer the basis
    :param filter_length: size of each basis element
    :param basis_size: num of elements in returned basis
    :param basis_type: basis type, can be 'chebyshev', 'classic' or 'classic_symmetric'
    :return: returns array of shape (basis_size, filter_length, filter_length) contains basis elements
    """
    if basis_type == 'chebyshev':
        return _create_chebyshev_basis(filter_length, basis_size)
    if basis_type == 'rings':
        return _create_rings_basis(filter_length, basis_size)

    raise Exception('Unsupported basis type was provided.')


def _create_rings_basis(filter_length, basis_size):
    """
    this basis contains 'basis_size' rings that cover circle of diamiter 'filter_length'
    :param filter_length: each element length
    :param basis_size: num of elements in returned basis
    """
    ring_width = filter_length // (basis_size * 2)
    basis = np.zeros(shape=(basis_size, filter_length, filter_length))
    temp_map = np.zeros(shape=(filter_length, filter_length))

    radius = filter_length // 2
    center = (radius, radius)
    for i in range(basis_size):
        rr, cc = disk(center, radius - i * ring_width)
        temp_map[rr, cc] = i + 1

    for i in range(basis_size):
        basis[i] = temp_map * (temp_map == i + 1) / (i + 1)

    return basis


def _create_chebyshev_basis(filter_length, basis_size):
    """
    creates basis contains even chebyshev terms for symmetric signals
    :param filter_length: each element length
    :param basis_size: num of elements in returned basis
    """
    basis = np.zeros(shape=(basis_size, filter_length, filter_length))
    center = (int(filter_length / 2), int(filter_length / 2))
    radius = min(center[0], center[1], filter_length - center[0], filter_length - center[1])

    Y, X = np.ogrid[:filter_length, :filter_length]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center > radius
    for i in range(basis_size):
        chebyshev_basis_element = np.polynomial.chebyshev.Chebyshev.basis(2 * i, [-radius, radius])
        basis_element = chebyshev_basis_element(dist_from_center)
        basis_element[mask] = 0
        basis[i] = basis_element
    gs_basis = gram_schmidt(basis.reshape(basis_size, -1)).reshape(basis_size, filter_length, filter_length)
    return gs_basis
