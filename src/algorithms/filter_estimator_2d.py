import numpy as np
from scipy.signal import convolve

from src.algorithms.utils import log_size_S_2d_1axis, calc_mapping_2d, \
    _calc_likelihood_and_likelihood_derivative_without_constants_2d, log_prob_all_is_noise, _gradient_descent


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

        if len(unnormalized_data.shape) == 1:
            unnormalized_data = np.array([unnormalized_data])

        self.unnormalized_data = unnormalized_data
        self.unnormalized_filter_basis = unnormalized_filter_basis
        self.num_of_instances = num_of_instances
        self.noise_std = noise_std

        self.data = (unnormalized_data - noise_mean) / noise_std
        self.filter_basis_size = len(unnormalized_filter_basis)
        self.filter_shape = unnormalized_filter_basis[0].shape
        self.filter_basis, self.basis_norms = self.normalize_basis()

        # self.convolved_basis, self.convolved_filter = self.calc_constants()
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
        return log_size_S_2d_1axis(self.data.shape[0], self.num_of_instances, self.filter_shape[0])

    def calc_term_three_const(self):
        """
        The 3rd term of the likelihood function
        """
        term = -self.num_of_instances / 2
        return term

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
                                    self.data.shape[0] - self.filter_shape + 1,
                                    self.data.shape[1] - self.filter_shape + 1))
        for i, fil in enumerate(self.filter_basis):
            constants[i] = self.convolve_basis_element(fil)
        return constants

    def calc_convolved_filter(self, sample_index, filter_coeffs):
        """
        :param sample_index: index of the sample among m data samples
        :param filter_coeffs: coefficients for the filter basis
        :return: the dot product between relevant data sample and filter
        """
        convolved_filter = np.inner(self.convolved_basis[sample_index].T, filter_coeffs)
        return convolved_filter

    def calc_mapping(self, convolved_filter):
        """
        :param convolved_filter: dot product between data sample and a filter
        :return: likelihood mapping
        """
        return calc_mapping_2d(self.data.shape[0],
                               self.num_of_instances,
                               self.filter_shape[0],
                               convolved_filter)

    def calc_gradient(self, sample_index, filter_coeffs, mapping=None):
        """
        calculate sample gradient with respect to each filter coefficient
        :param sample_index: index of the data sample
        :param filter_coeffs: filter coefficients
        :param mapping: likelihood mapping
        :return: gradient of the filter coefficients
        """
        convolved_filter = self.calc_convolved_filter(sample_index, filter_coeffs)

        if mapping is None:
            mapping = self.calc_mapping(convolved_filter)

        gradient = np.zeros(self.filter_basis_size)
        for i in range(self.filter_basis_size):
            gradient[i] = _calc_term_two_derivative_1d(self.sample_length,
                                                       self.num_of_instances,
                                                       self.filter_shape,
                                                       self.convolved_basis[sample_index][i],
                                                       convolved_filter,
                                                       mapping)
        return gradient + 2 * self.term_three_const * filter_coeffs

    def calc_likelihood_and_gradient(self, filter_coeffs):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """
        likelihoods = np.zeros(self.num_of_data_samples)
        gradients = np.zeros(shape=(self.num_of_data_samples, self.filter_basis_size))

        for i, sample in enumerate(self.data):
            convolved_filter = self.calc_convolved_filter(i, filter_coeffs)
            mapping = self.calc_mapping(convolved_filter)
            gradients[i] = self.calc_gradient(i, filter_coeffs, mapping)
            likelihoods[i] = self.term_one + \
                             self.term_two + \
                             self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + \
                             mapping[0, self.num_of_instances]

        # return likelihoods.mean(axis=0), gradients.mean(axis=0)
        return likelihoods.sum(axis=0), gradients.sum(axis=0)

    def estimate(self):
        """
        Estimate optimal the match filter using given data samples and with respect to given filter basis
        :return: likelihood value and optimal unnormalized filter coefficient (can be used on user basis)
        """

        def _calc_likelihood_and_gradient(filter_coeffs):
            return self.calc_likelihood_and_gradient(filter_coeffs)

        initial_coeffs, t, epsilon, max_iter = np.zeros(self.filter_basis_size), 0.1, 1e-2, 100
        # l, initial_coeffs = self.estimate_max()
        initial_coeffs = np.array([1 - np.square(i) for i in np.linspace(-1, 1, self.filter_basis_size)])
        likelihood, normalized_optimal_coeffs = _gradient_descent(_calc_likelihood_and_gradient, initial_coeffs, t,
                                                                  epsilon, max_iter, concave=True)

        optimal_coeffs = normalized_optimal_coeffs * self.noise_std / self.basis_norms
        return likelihood, optimal_coeffs