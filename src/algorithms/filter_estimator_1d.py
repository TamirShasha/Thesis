import numpy as np
from scipy.signal import convolve
from time import time

from src.algorithms.utils import log_size_S_1d, calc_mapping_1d, _calc_term_two_derivative_1d, \
    log_prob_all_is_noise, _gradient_descent, _calc_mapping_1d_many, gram_schmidt


class FilterEstimator1D:

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
        :param num_of_instances: estimation of signal num of instances inside given data, for array kind of data, this number is fixed for all samples
        """

        if len(unnormalized_filter_basis) == 0:
            raise Exception('Must provide basis')

        if len(unnormalized_data.shape) == 1:
            unnormalized_data = np.array([unnormalized_data])

        self.unnormalized_data = unnormalized_data
        self.unnormalized_filter_basis = unnormalized_filter_basis
        self.num_of_instances = num_of_instances
        self.noise_std = noise_std

        self.data = (unnormalized_data - noise_mean) / noise_std
        self.num_of_data_samples = unnormalized_data.shape[0]
        self.sample_length = unnormalized_data.shape[1]
        self.filter_basis_size = len(unnormalized_filter_basis)
        self.filter_length = len(unnormalized_filter_basis[0])
        self.filter_basis, self.basis_norms = self.normalize_basis()

        self.convolved_basis = self.convolve_basis()
        self.term_one = -self.calc_log_size_s()
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

    def calc_log_size_s(self):
        return log_size_S_1d(self.sample_length, self.num_of_instances, self.filter_length)

    def calc_term_three_const(self):
        """
        The 3rd term of the likelihood function
        """
        term = -self.num_of_instances / 2
        return term

    def convolve_basis_element(self, filter_element):
        """
        convolve one basis element with each of the data samples
        :param filter_element: one element from filter basis
        :return: (T, n-d) while T is basis size, n is data length and d is filter length
        """
        flipped_signal_filter = np.flip(filter_element)  # Flipping to cross-correlate
        conv = np.array([convolve(self.data[i], flipped_signal_filter, mode='valid')
                         for i in range(self.num_of_data_samples)])
        return conv

    def convolve_basis(self):
        """
        convolve each normalized basis element with data
        :return: (m, T, n-d) array where T is basis size, n is data length, d is filter length and m is num of samples
        """
        constants = np.zeros(shape=(self.num_of_data_samples,
                                    self.filter_basis_size,
                                    self.sample_length - self.filter_length + 1))
        for i, fil in enumerate(self.filter_basis):
            constants[:, i, :] = self.convolve_basis_element(fil)
        return constants

    def calc_convolved_filter(self, sample_index, filter_coeffs):
        """
        :param sample_index: index of the sample among m data samples
        :param filter_coeffs: coefficients for the filter basis
        :return: the dot product between relevant data sample and filter
        """
        return np.inner(self.convolved_basis[sample_index].T, filter_coeffs)

    def calc_mapping(self, convolved_filter):
        """
        :param convolved_filter: dot product between data sample and a filter
        :return: likelihood mapping
        """
        return calc_mapping_1d(self.sample_length,
                               self.num_of_instances,
                               self.filter_length,
                               convolved_filter)

    def calc_mappings(self, convolved_filters):
        """
        :param convolved_filters: array of dot product between data sample and a filter
        :return: likelihood mapping
        """
        return _calc_mapping_1d_many(self.sample_length,
                                     self.num_of_instances,
                                     self.filter_length,
                                     convolved_filters)

    def calc_gradient(self, sample_index, filter_coeffs, convolved_filter, mapping=None):
        """
        calculate sample gradient with respect to each filter coefficient
        :param sample_index: index of the data sample
        :param filter_coeffs: filter coefficients
        :param convolved_filter: convolution of filter with data sample
        :param mapping: likelihood mapping
        :return: gradient of the filter coefficients
        """
        if mapping is None:
            mapping = self.calc_mapping(convolved_filter)

        gradient = np.zeros(self.filter_basis_size)
        for i in range(self.filter_basis_size):
            gradient[i] = _calc_term_two_derivative_1d(self.sample_length,
                                                       self.num_of_instances,
                                                       self.filter_length,
                                                       self.convolved_basis[sample_index][i],
                                                       convolved_filter,
                                                       mapping)
        return gradient + 2 * self.term_three_const * filter_coeffs

    def calc_gradient_discrete(self, sample_index, filter_coeffs, mapping=None):
        """
        calculate sample gradient with respect to each filter coefficient
        :param sample_index: index of the data sample
        :param filter_coeffs: filter coefficients
        :param convolved_filter: convolution of filter with data sample
        :param mapping: likelihood mapping
        :return: gradient of the filter coefficients
        """
        likelihood = self.term_one + \
                     self.term_two + \
                     self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + \
                     mapping[0, -1]

        eps = 1e-4
        gradient = np.zeros(self.filter_basis_size)
        for i in range(self.filter_basis_size):
            filter_coeffs_perturbation = filter_coeffs + np.eye(1, self.filter_basis_size, i)[0] * eps
            convolved_filter = self.calc_convolved_filter(sample_index, filter_coeffs_perturbation)
            likelihood_perturbation = self.calc_mapping(convolved_filter)[0, -1] + \
                                      self.term_one + \
                                      self.term_two + \
                                      self.term_three_const * np.inner(filter_coeffs_perturbation,
                                                                       filter_coeffs_perturbation)
            gradient[i] = (likelihood_perturbation - likelihood) / eps

        return gradient

    def calc_likelihood_and_gradient(self, filter_coeffs):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """
        likelihoods = np.zeros(self.num_of_data_samples)
        gradients = np.zeros(shape=(self.num_of_data_samples, self.filter_basis_size))

        for i, sample in enumerate(self.data):
            convolved_filter = self.calc_convolved_filter(i, filter_coeffs)
            mapping = self.calc_mapping(convolved_filter)
            gradients[i] = self.calc_gradient_discrete(i, filter_coeffs, mapping)
            likelihoods[i] = self.term_one + \
                             self.term_two + \
                             self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + \
                             mapping[0, self.num_of_instances]

        return likelihoods.sum(axis=0), gradients.sum(axis=0)

    def calc_likelihood_and_gradient_experimental(self, filter_coeffs, eps=1e-4):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """
        t_start = time()
        likelihoods = np.zeros(self.num_of_data_samples)
        gradients = np.zeros(shape=(self.num_of_data_samples, self.filter_basis_size))

        convolved_filters = np.zeros(shape=(self.num_of_data_samples,
                                            self.filter_basis_size + 1,
                                            self.sample_length - self.filter_length + 1))
        # TODO: do all with 1 command
        filter_coeffs_with_perturbations = np.array([filter_coeffs] +
                                                    [filter_coeffs + np.eye(1, self.filter_basis_size, i)[0] * eps
                                                     for i in range(self.filter_basis_size)])
        for i, sample in enumerate(self.data):
            for t in range(self.filter_basis_size + 1):
                convolved_filters[i, t] = self.calc_convolved_filter(i, filter_coeffs_with_perturbations[t])

        t0 = time()
        mappings = self.calc_mappings(convolved_filters.reshape(-1, self.sample_length - self.filter_length + 1))
        print(f'took {time() - t0} seconds')
        mappings = mappings[:, 0, self.num_of_instances].reshape(self.num_of_data_samples, self.filter_basis_size + 1)

        for i in range(self.num_of_data_samples):
            term_three = self.term_three_const * np.array([np.inner(fc, fc) for fc in filter_coeffs_with_perturbations])
            sample_likelihoods = self.term_one + \
                                 self.term_two + \
                                 term_three + \
                                 mappings[i, :]
            gradients[i] = (sample_likelihoods[1:] - sample_likelihoods[0]) / eps
            likelihoods[i] = sample_likelihoods[0]

        # return likelihoods.mean(axis=0), gradients.mean(axis=0)
        print(f'all took {time() - t_start} seconds')
        return likelihoods.sum(axis=0), gradients.sum(axis=0)

    def estimate(self):
        """
        Estimate optimal the match filter using given data samples and with respect to given filter basis
        :return: likelihood value and optimal unnormalized filter coefficient (can be used on user basis)
        """

        def _calc_likelihood_and_gradient(filter_coeffs):
            return self.calc_likelihood_and_gradient(filter_coeffs)

        initial_coeffs, t, epsilon, max_iter = np.zeros(self.filter_basis_size), 0.1, 1e-4, 100
        likelihood, normalized_optimal_coeffs = _gradient_descent(_calc_likelihood_and_gradient, initial_coeffs, t,
                                                                  epsilon, max_iter, concave=True)

        optimal_coeffs = normalized_optimal_coeffs * self.noise_std / self.basis_norms
        return likelihood, optimal_coeffs

    # def estimate_max(self):
    #
    #     _p = np.zeros(self.filter_basis_size)
    #     likelihoods_pos = []
    #     likelihoods_neg = []
    #     p_pos = []
    #     p_neg = []
    #     for i in range(self.num_of_data_samples):
    #         print(i)
    #         c = self.convolved_basis[i].T
    #         likelihood_pos, locations_pos = heuristic_dp(self.sample_length, self.filter_length, self.num_of_instances,
    #                                                      c)
    #         likelihood_neg, locations_neg = heuristic_dp(self.sample_length, self.filter_length, self.num_of_instances,
    #                                                      -c)
    #
    #         p_pos.append(np.sum(c[locations_pos], 0) / (2 * self.term_three_const))
    #         p_neg.append(np.sum(-c[locations_neg], 0) / (2 * self.term_three_const))
    #         likelihoods_pos.append(likelihood_pos)
    #         likelihoods_neg.append(likelihood_neg)
    #
    #     if likelihood_pos > likelihood_neg:
    #         p = np.nanmean(p_pos, 0)
    #     else:
    #         p = np.nanmean(p_neg, 0)
    #
    #     optimal_coeffs = p * self.noise_std / self.basis_norms
    #     return 0, optimal_coeffs


def create_filter_basis(filter_length, basis_size, basis_type='chebyshev'):
    """
    creates basis for signals of given size
    the higher the size, the finer the basis
    :param filter_length: size of each basis element
    :param basis_size: num of elements in returned basis
    :param basis_type: basis type, can be 'chebyshev', 'classic' or 'classic_symmetric'
    :return: returns array of shape (basis_size, filter_length) contains basis elements
    """
    if basis_type == 'chebyshev':
        return _create_chebyshev_basis(filter_length, basis_size)
    if basis_type == 'classic_symmetric':
        return _create_classic_symmetric_basis(filter_length, basis_size)
    if basis_size == 'classic':
        return _create_classic_basis(filter_length, basis_size)

    raise Exception('Unsupported basis type was provided.')


def _create_classic_symmetric_basis(filter_length, basis_size):
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


def _create_classic_basis(filter_length, basis_size):
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


def _create_chebyshev_basis(filter_length, basis_size):
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
