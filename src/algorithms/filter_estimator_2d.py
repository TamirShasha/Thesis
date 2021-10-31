import numpy as np
from scipy.signal import convolve
from skimage.draw import disk

from src.algorithms.utils import log_size_S_2d_1axis, calc_mapping_2d, log_prob_all_is_noise, \
    _gradient_descent, gram_schmidt
from src.utils.logsumexp import logsumexp_simple


class FilterEstimator2D:

    def __init__(self,
                 unnormalized_data: np.ndarray,
                 unnormalized_filter_basis: np.ndarray,
                 num_of_instances: int,
                 noise_std=1.,
                 noise_mean=0.):
        """
        initialize the filter estimator
        :param unnormalized_data: 1d data
        :param unnormalized_filter_basis: list of orthonormal basis
        """

        if len(unnormalized_filter_basis) == 0:
            raise Exception('Must provide basis')

        self.unnormalized_data = unnormalized_data
        self.unnormalized_filter_basis = unnormalized_filter_basis
        self.num_of_instances = num_of_instances
        self.noise_std = noise_std

        self.data = (unnormalized_data - noise_mean) / noise_std
        self.filter_basis_size = len(unnormalized_filter_basis)
        self.filter_shape = unnormalized_filter_basis[0].shape
        self.filter_basis, self.basis_norms = self.normalize_basis()

        # find max possible instances for given filter size
        self.max_possible_instances = min(
            self.num_of_instances,
            (self.data.shape[0] // self.filter_shape[0]) * (self.data.shape[1] // self.filter_shape[1]))

        self.convolved_basis = self.convolve_basis()

        self.term_one = -logsumexp_simple(
            np.array([self.calc_log_size_s(k) for k in range(self.max_possible_instances)]))
        self.term_two = log_prob_all_is_noise(self.data, 1)
        self.term_three_const = self.calc_term_three_const()

    def normalize_basis(self):
        """
        :return: normalized basis and respected norms
        """
        normalized_basis = np.zeros_like(self.unnormalized_filter_basis)
        basis_norms = np.zeros(self.filter_basis_size)
        for i, fil in enumerate(self.unnormalized_filter_basis):
            norm = np.linalg.norm(fil)
            basis_norms[i] = norm
            normalized_basis[i] = fil / norm

        return normalized_basis, basis_norms

    def calc_log_size_s(self, k):
        return log_size_S_2d_1axis(self.data.shape[0], k, self.filter_shape[0])

    def calc_term_three_const(self):
        """
        The 3rd term of the likelihood function
        """
        return -np.arange(1, self.max_possible_instances + 1) / 2
        # term = -self.num_of_instances / 2
        # return term

    def convolve_basis_element(self, filter_element):
        """
        convolve one basis element with data
        :param filter_element: one element from filter basis
        :return: (n-d, m-d) while  n is data #rows, m is #columns
        """
        flipped_signal_filter = np.flip(filter_element)  # Flipping to cross-correlate
        conv = convolve(self.data, flipped_signal_filter, mode='valid')
        return conv

    def convolve_basis(self):
        """
        convolve each normalized basis element with data
        :return: (T, n-d, m-d) array where T is basis size, n is #rows, m is #columns
        """
        constants = np.zeros(shape=(self.filter_basis_size,
                                    self.data.shape[0] - self.filter_shape[0] + 1,
                                    self.data.shape[1] - self.filter_shape[1] + 1))
        for i, fil in enumerate(self.filter_basis):
            constants[i] = self.convolve_basis_element(fil)
        return constants

    def calc_convolved_filter(self, filter_coeffs):
        """
        :param filter_coeffs: coefficients for the filter basis
        :return: the dot product between relevant data sample and filter
        """
        convolved_filter = np.inner(self.convolved_basis.T, filter_coeffs)
        return convolved_filter

    def calc_mapping(self, convolved_filter):
        """
        :param convolved_filter: dot product between data sample and a filter
        :return: likelihood mapping
        """
        return calc_mapping_2d(self.data.shape[0],
                               self.max_possible_instances,
                               self.filter_shape[0],
                               convolved_filter)

    def calc_likelihood(self, filter_coeffs, mapping):
        term1 = -logsumexp_simple(
            np.array([log_size_S_2d_1axis(self.data.shape[0], k + 1, self.filter_shape[0]) for k in
                      range(self.max_possible_instances)]))
        term2 = self.term_two
        term3 = logsumexp_simple(self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + mapping[0, 1:])

        likelihood = term1 + term2 + term3
        return likelihood

    def calc_gradient_discrete(self, filter_coeffs, likelihood):
        """
        calculate gradient with respect to each filter coefficient
        :param filter_coeffs: current point filter coefficients
        :param likelihood: current point likelihood
        :return: gradient of the filter coefficients (current point)
        """

        eps = 1e-4
        gradient = np.zeros(self.filter_basis_size)
        for i in range(self.filter_basis_size):
            filter_coeffs_perturbation = filter_coeffs + np.eye(1, self.filter_basis_size, i)[0] * eps
            convolved_filter = self.calc_convolved_filter(filter_coeffs_perturbation)
            mapping = self.calc_mapping(convolved_filter)

            likelihood_perturbation = self.calc_likelihood(filter_coeffs_perturbation, mapping)
            # likelihood_perturbation = mapping[0, -1] + \
            #                           self.term_one + \
            #                           self.term_two + \
            #                           self.term_three_const * np.inner(filter_coeffs_perturbation,
            #                                                            filter_coeffs_perturbation)
            gradient[i] = (likelihood_perturbation - likelihood) / eps

        return gradient

    def calc_likelihood_and_gradient(self, filter_coeffs):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """

        convolved_filter = self.calc_convolved_filter(filter_coeffs)
        mapping = self.calc_mapping(convolved_filter)

        # import matplotlib.pyplot as plt
        # term_ones = np.array([log_size_S_2d_1axis(self.data.shape[0], k + 1, self.filter_shape[0]) for k in
        #                       range(self.max_possible_instances)])
        # term_threes = -np.arange(1, self.max_possible_instances + 1) / 2
        # likelihoods = -term_ones + self.term_two + term_threes * np.inner(filter_coeffs, filter_coeffs) + mapping[0, 1:]
        # plt.plot(likelihoods)
        # plt.show()

        # likelihood = self.term_one + \
        #              self.term_two + \
        #              self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + \
        #              mapping[0, self.num_of_instances]
        likelihood = self.calc_likelihood(filter_coeffs, mapping)
        gradient = self.calc_gradient_discrete(filter_coeffs, likelihood)

        return likelihood, gradient

    def estimate(self):
        """
        Estimate optimal the match filter using given data samples and with respect to given filter basis
        :return: likelihood value and optimal unnormalized filter coefficient (can be used on user basis)
        """

        # if self.term_one == np.inf:
        #     return -np.inf, np.zeros(self.filter_basis_size)

        initial_coeffs, t, epsilon, max_iter = np.zeros(self.filter_basis_size), 0.1, 1e-2, 20
        likelihood, normalized_optimal_coeffs = _gradient_descent(self.calc_likelihood_and_gradient, initial_coeffs, t,
                                                                  epsilon, max_iter, concave=True)

        optimal_coeffs = normalized_optimal_coeffs * self.noise_std / self.basis_norms
        return likelihood, optimal_coeffs


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
