import numpy as np
from scipy.signal import convolve

from src.algorithms.utils import log_size_S_1d, calc_mapping_1d, _calc_term_two_derivative_1d, log_prob_all_is_noise, \
    _gradient_descent
from src.experimental.dev3 import heuristic_dp


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
        convolved_filter = np.inner(self.convolved_basis[sample_index].T, filter_coeffs)
        return convolved_filter

    def calc_mapping(self, convolved_filter):
        """
        :param convolved_filter: dot product between data sample and a filter
        :return: likelihood mapping
        """
        return calc_mapping_1d(self.sample_length,
                               self.num_of_instances,
                               self.filter_length,
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
                                                       self.filter_length,
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

    def estimate_max(self):

        _p = np.zeros(self.filter_basis_size)
        likelihoods_pos = []
        likelihoods_neg = []
        p_pos = []
        p_neg = []
        for i in range(self.num_of_data_samples):
            print(i)
            c = self.convolved_basis[i].T
            likelihood_pos, locations_pos = heuristic_dp(self.sample_length, self.filter_length, self.num_of_instances,
                                                         c)
            likelihood_neg, locations_neg = heuristic_dp(self.sample_length, self.filter_length, self.num_of_instances,
                                                         -c)

            # if likelihood_pos > likelihood_neg:
            #     likelihood = likelihood_pos
            #     locations = locations_pos
            #
            # else:
            #     c = -c
            #     likelihood = likelihood_neg
            #     locations = locations_neg

            p_pos.append(np.sum(c[locations_pos], 0) / (2 * self.term_three_const))
            p_neg.append(np.sum(-c[locations_neg], 0) / (2 * self.term_three_const))
            likelihoods_pos.append(likelihood_pos)
            likelihoods_neg.append(likelihood_neg)

            # p = np.sum(c[locations], 0) / (2 * self.term_three_const)
            # _p += p

        likelihood_pos = sum(likelihoods_pos)
        likelihood_neg = sum(likelihoods_neg)

        if likelihood_pos > likelihood_neg:
            p = np.nanmean(p_pos, 0)
        else:
            p = np.nanmean(p_neg, 0)

        # _p /= self.num_of_data_samples
        optimal_coeffs = p * self.noise_std / self.basis_norms
        return 0, p


def create_symmetric_basis(length, size):
    """
    creates basis for symmetric signals of given size
    the higher the size, the finer the basis
    :param length: each element length
    :param size: num of elements in returned basis
    """
    step_width = length // (size * 2)
    basis = np.zeros(shape=(size, length))

    for i in range(size):
        pos = i * step_width
        basis[i, pos: pos + step_width] = 1

    basis += np.flip(basis, axis=1)
    basis[size - 1, size * step_width:size * step_width + length % (size * 2)] = 1

    return basis


def create_span_basis(length, size):
    """
    creates basis for discrete signals of given size
    the higher the size, the finer the basis
    :param length: each element length
    :param size: num of elements in returned basis
    """
    step_width = length // size
    basis = np.zeros(shape=(size, length))

    end = 0
    for i in range(size):
        start = end
        end = start + step_width
        if i < length % size:
            end += 1
        basis[i, start: end] = 1

    return basis


from src.experiments.data_simulator_1d import simulate_data
import matplotlib.pyplot as plt


# b = create_span_basis(19, 10)
# print(b)
# for a in b:
#     plt.plot(a)
# plt.show()


def exp():
    n, d, p, k, sig = 6000, 100, 1, 10, .2
    M = 10

    # filter_basis = create_symmetric_basis(100, 7)
    # print(np.matmul(filter_basis, filter_basis.T))
    # print(filter_basis)
    # plt.plot(filter_basis)
    # plt.show()

    # signal = np.concatenate([np.full(5, 1 / 3),
    #                          np.full(5, -2 / 3),
    #                          np.full(5, -1),
    #                          np.full(5, 2 / 3),
    #                          np.full(5, 2 / 3),
    #                          np.full(5, -1),
    #                          np.full(5, -2 / 3),
    #                          np.full(5, 1 / 3)])
    signal = np.array([1 - np.square(i) for i in np.linspace(-1, 1, d)])
    # signal = np.concatenate([np.full(d//2, 1),
    #                          np.full(d//2, 1/2)])

    # print(np.linalg.norm(signal))

    data = []
    for i in range(M):
        _data, pulses = simulate_data(n, d, p, k, sig, signal)
        data.append(_data)
    data = np.array(data)
    # plt.plot(data)
    # plt.show()

    # filter_basis = np.array([
    #     np.concatenate([np.ones(15), np.zeros(15)]),
    #     np.concatenate([np.zeros(15), np.ones(15)])
    # ])

    plt.plot(signal)
    plt.show()
    # filter_basis = np.array([
    #     np.concatenate([np.ones(5), np.zeros(30), np.ones(5)]),
    #     np.concatenate([np.zeros(5), np.ones(5), np.zeros(20), np.ones(5), np.zeros(5)]),
    #     np.concatenate([np.zeros(10), np.ones(5), np.zeros(10), np.ones(5), np.zeros(10)]),
    #     np.concatenate([np.zeros(15), np.ones(10), np.zeros(15)]),
    # ])

    # [plt.plot(fil) for fil in filter_basis]
    # plt.show()

    likelihoods = []
    for _d in np.arange(100, 1001, 100):
        # filter_basis = create_symmetric_basis(_d, 7)
        filter_basis = create_span_basis(_d, 10)
        # plt.plot(filter_basis)
        # plt.show()
        # filter_basis = create_span_basis(_d, 10)
        fil_est = FilterEstimator1D(data, filter_basis, k//2, sig)
        # f, p = fil_est.estimate_max()
        f, p = fil_est.estimate()
        likelihoods.append(f)

        plt.plot(filter_basis.T.dot(p))
        plt.plot(signal)
        plt.show()

    plt.plot(likelihoods)
    plt.show()


# exp()
