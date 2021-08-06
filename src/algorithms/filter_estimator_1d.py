import numpy as np
from scipy.signal import convolve

from src.algorithms.utils import log_size_S_1d, calc_mapping_1d, _calc_term_two_derivative_1d, log_prob_all_is_noise, \
    _gradient_descent


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
        term = -self.num_of_instances / 2
        return term

    def convolve_basis_element(self, filter_element):
        """
        :param filter_element: one element from filter basis
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
        convolved_filter = np.inner(self.convolved_basis[sample_index].T, filter_coeffs)
        return convolved_filter

    def calc_mapping(self, convolved_filter):
        return calc_mapping_1d(self.sample_length,
                               self.num_of_instances,
                               self.filter_length,
                               convolved_filter)

    def calc_gradient(self, sample_index, filter_coeffs, mapping=None):
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

        return likelihoods.mean(axis=0), gradients.mean(axis=0)

    def estimate(self):
        def _calc_likelihood_and_gradient(filter_coeffs):
            return self.calc_likelihood_and_gradient(filter_coeffs)

        initial_coeffs, t, epsilon, max_iter = self.basis_norms, 0.1, 1e-4, 100
        likelihood, normalized_optimal_coeffs = _gradient_descent(_calc_likelihood_and_gradient, initial_coeffs, t,
                                                                  epsilon, max_iter, concave=True)

        optimal_coeffs = normalized_optimal_coeffs * self.noise_std / self.basis_norms
        return likelihood, optimal_coeffs


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


from src.experiments.data_simulator_1d import simulate_data
import matplotlib.pyplot as plt
from src.algorithms.utils import calc_most_likelihood_and_optimized_power_1d


def exp():
    n, d, p, k, sig = 10000, 40, 1, 100, 1

    # signal = np.concatenate([np.full(5, 1 / 3),
    #                          np.full(5, -2 / 3),
    #                          np.full(5, -1),
    #                          np.full(5, 2 / 3),
    #                          np.full(5, 2 / 3),
    #                          np.full(5, -1),
    #                          np.full(5, -2 / 3),
    #                          np.full(5, 1 / 3)])
    signal = np.array([1 - np.square(i) for i in np.linspace(-1, 1, 40)])

    print(np.linalg.norm(signal))

    data, pulses = simulate_data(n, d, p, k, sig, signal)
    data2, pulses = simulate_data(n, d, p, k, sig, signal)
    # plt.plot(data)
    # plt.show()

    # filter_basis = np.array([
    #     np.concatenate([np.ones(15), np.zeros(15)]),
    #     np.concatenate([np.zeros(15), np.ones(15)])
    # ])

    # plt.plot(signal)
    # plt.show()
    # filter_basis = np.array([
    #     np.concatenate([np.ones(5), np.zeros(30), np.ones(5)]),
    #     np.concatenate([np.zeros(5), np.ones(5), np.zeros(20), np.ones(5), np.zeros(5)]),
    #     np.concatenate([np.zeros(10), np.ones(5), np.zeros(10), np.ones(5), np.zeros(10)]),
    #     np.concatenate([np.zeros(15), np.ones(10), np.zeros(15)]),
    # ])

    # [plt.plot(fil) for fil in filter_basis]
    # plt.show()

    filter_basis = create_symmetric_basis(2 * d, 10)
    fil_est = FilterEstimator1D(np.array([data, data2]), filter_basis, k, sig)
    f, p = fil_est.estimate()
    print(p)

    plt.plot(filter_basis.T.dot(p))
    plt.plot(signal)
    plt.show()


# exp()
