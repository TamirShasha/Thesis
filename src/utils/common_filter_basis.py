from skimage.draw import disk
import numpy as np
from scipy.stats import norm

from src.algorithms.utils import gram_schmidt


def create_filter_basis_1d(filter_length, basis_size, basis_type='chebyshev'):
    """
    creates basis for signals of given size
    the higher the size, the finer the basis
    :param filter_length: size of each basis element
    :param basis_size: num of elements in returned basis
    :param basis_type: basis type, can be 'chebyshev', 'classic' or 'classic_symmetric'
    :return: returns array of shape (basis_size, filter_length) contains basis elements
    """
    if basis_type == 'chebyshev':
        return _create_chebyshev_basis_1d(filter_length, basis_size)
    if basis_type == 'classic_symmetric':
        return _create_classic_symmetric_basis_1d(filter_length, basis_size)
    if basis_size == 'classic':
        return _create_classic_basis_1d(filter_length, basis_size)

    raise Exception('Unsupported basis type was provided.')


def _create_classic_symmetric_basis_1d(filter_length, basis_size):
    """
    creates basis for symmetric signals of given size
    the higher the size, the finer the basis
    :param filter_length: each element length
    :param basis_size: num of elements in returned basis
    """
    step_width = filter_length // (basis_size * 2)
    basis = np.zeros(shape=(basis_size, filter_length))

    for i in range(basis_size):
        pos = i * step_width
        basis[i, pos: pos + step_width] = 1

    basis += np.flip(basis, axis=1)
    basis[basis_size - 1, basis_size * step_width:basis_size * step_width + filter_length % (basis_size * 2)] = 1

    return basis


def _create_classic_basis_1d(filter_length, basis_size):
    """
    creates basis for discrete signals of given size
    the higher the size, the finer the basis
    :param filter_length: each element length
    :param basis_size: num of elements in returned basis
    """
    step_width = filter_length // basis_size
    basis = np.zeros(shape=(basis_size, filter_length))

    end = 0
    for i in range(basis_size):
        start = end
        end = start + step_width
        if i < filter_length % basis_size:
            end += 1
        basis[i, start: end] = 1

    return basis


def _create_chebyshev_basis_1d(filter_length, basis_size):
    """
    creates basis contains chebyshev terms
    :param filter_length: each element length
    :param basis_size: num of elements in returned basis
    """
    basis = np.zeros(shape=(basis_size, filter_length))
    xs = np.linspace(-1, 1, filter_length)
    for i in range(basis_size):
        chebyshev_basis_element = np.polynomial.chebyshev.Chebyshev.basis(i)
        basis[i] = chebyshev_basis_element(xs)
    basis = gram_schmidt(basis)
    return basis


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
    if basis_type == 'gaussian':
        return _create_gaussian_basis(filter_length, basis_size)

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


def _create_gaussian_basis(filter_length, basis_size):
    basis = np.zeros(shape=(basis_size, filter_length, filter_length))
    center = (int(filter_length / 2), int(filter_length / 2))
    radius = min(center[0], center[1], filter_length - center[0], filter_length - center[1])

    Y, X = np.ogrid[:filter_length, :filter_length]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center > radius
    for i in range(basis_size):
        basis_element = norm.pdf(dist_from_center, 0, filter_length / (2 * (basis_size - i)))
        basis_element[mask] = 0
        basis[i] = basis_element
    gs_basis = gram_schmidt(basis.reshape(basis_size, -1)).reshape(basis_size, filter_length, filter_length)
    return gs_basis
