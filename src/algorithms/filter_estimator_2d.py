import numpy as np
from scipy.signal import convolve
from skimage.draw import disk

from src.algorithms.utils import log_size_S_2d_1axis, calc_mapping_2d, \
    _calc_likelihood_and_likelihood_derivative_without_constants_2d, log_prob_all_is_noise, _gradient_descent
from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D


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
        convolved_filter = self.calc_convolved_filter(filter_coeffs)

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

    def calc_gradient_discrete(self, filter_coeffs, likelihood):
        """
        calculate sample gradient with respect to each filter coefficient
        :param filter_coeffs: filter coefficients
        :param convolved_filter: convolution of filter with data sample
        :param mapping: likelihood mapping
        :return: gradient of the filter coefficients
        """

        eps = 1e-4
        gradient = np.zeros(self.filter_basis_size)
        for i in range(self.filter_basis_size):
            filter_coeffs_perturbation = filter_coeffs + np.eye(1, self.filter_basis_size, i)[0] * eps
            convolved_filter = self.calc_convolved_filter(filter_coeffs_perturbation)
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

        convolved_filter = self.calc_convolved_filter(filter_coeffs)
        mapping = self.calc_mapping(convolved_filter)
        likelihood = self.term_one + \
                     self.term_two + \
                     self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + \
                     mapping[0, self.num_of_instances]
        gradient = self.calc_gradient_discrete(filter_coeffs, likelihood)

        return likelihood, gradient

    def estimate(self):
        """
        Estimate optimal the match filter using given data samples and with respect to given filter basis
        :return: likelihood value and optimal unnormalized filter coefficient (can be used on user basis)
        """

        initial_coeffs, t, epsilon, max_iter = np.zeros(self.filter_basis_size), 0.1, 1e-2, 100
        likelihood, normalized_optimal_coeffs = _gradient_descent(self.calc_likelihood_and_gradient, initial_coeffs, t,
                                                                  epsilon, max_iter, concave=True)

        optimal_coeffs = normalized_optimal_coeffs * self.noise_std / self.basis_norms
        return likelihood, optimal_coeffs


def create_basis(diamiter, dim):
    ring_width = diamiter // (dim * 2)
    basis = np.zeros(shape=(dim, diamiter, diamiter))
    temp_map = np.zeros(shape=(diamiter, diamiter))

    radius = diamiter // 2
    center = (radius, radius)
    for i in range(dim):
        rr, cc = disk(center, radius - i * ring_width)
        temp_map[rr, cc] = i + 1

    for i in range(dim):
        basis[i] = temp_map * (temp_map == i + 1) / (i + 1)

    return basis


import matplotlib.pyplot as plt


def plot_signal_3d(signal):
    from matplotlib import cm

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(signal.shape[0])
    Y = np.arange(signal.shape[1])
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    # surf = ax.plot_surface(X, Y, signal,
    #                        linewidth=0, antialiased=False)
    #
    # # Customize the z axis.
    # # ax.set_zlim(-1.01, 1.01)
    # # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')
    #
    # plt.show()

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, signal, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def calc_error(signal, est_signal):
    return np.linalg.norm(signal - est_signal)


def exp():
    rows = 1500
    columns = 1500
    signal_length = 100
    signal_power = 1
    signal_fraction = 1 / 8
    # signal_gen = lambda l, p: Shapes2D.double_disk(l, l // 2, -p // 2, p)
    signal_gen = Shapes2D.sphere
    noise_std = 5
    noise_mean = 0
    apply_ctf = False

    sim_data = DataSimulator2D(rows=rows,
                               columns=columns,
                               signal_length=signal_length,
                               signal_power=signal_power,
                               signal_fraction=signal_fraction,
                               signal_gen=signal_gen,
                               noise_std=noise_std,
                               noise_mean=noise_mean,
                               apply_ctf=apply_ctf)

    data = sim_data.simulate()
    signal = sim_data.create_signal_instance()
    k = sim_data.occurrences

    plt.imshow(signal)
    plt.colorbar()
    plt.show()
    plt.imshow(data, cmap='gray')
    plt.show()

    filter_basis = create_basis(100, 5)
    filter_estimator = FilterEstimator2D(data, filter_basis, 20, noise_std)
    likelihood, optimal_coeffs = filter_estimator.estimate()
    est_signal = filter_basis.T.dot(optimal_coeffs)

    print(optimal_coeffs)
    err = calc_error(signal, est_signal)
    plt.title(f'({rows}, {columns}), length={signal_length}, k={k}, std={noise_std}, \nerror={np.round(err, 3)}')
    plt.imshow(est_signal)
    plt.colorbar()
    plt.show()

    # plot_signal_3d(est_signal)


def exp2():
    rows = 1500
    columns = 1500
    signal_length = 100
    signal_power = 1
    signal_fraction = 1 / 8
    # signal_gen = lambda l, p: Shapes2D.double_disk(l, l // 2, -p // 2, p)
    signal_gen = Shapes2D.sphere
    noise_std = 5
    noise_mean = 0
    apply_ctf = False

    sim_data = DataSimulator2D(rows=rows,
                               columns=columns,
                               signal_length=signal_length,
                               signal_power=signal_power,
                               signal_fraction=signal_fraction,
                               signal_gen=signal_gen,
                               noise_std=noise_std,
                               noise_mean=noise_mean,
                               apply_ctf=apply_ctf)

    data = sim_data.simulate()
    signal = sim_data.create_signal_instance()
    k = sim_data.occurrences

    plt.imshow(signal)
    plt.colorbar()
    plt.show()
    plt.imshow(data, cmap='gray')
    plt.show()

    filter_basis = create_basis(100, 5)

    ks = [1, 5, 10, 20, 35, 50, 70]
    # ks = [1, 5]
    errors = np.zeros_like(ks, dtype=float)
    for i, _k in enumerate(ks):
        filter_estimator = FilterEstimator2D(data, filter_basis, _k)
        likelihood, optimal_coeffs = filter_estimator.estimate()
        est_signal = filter_basis.T.dot(optimal_coeffs)
        err = calc_error(signal, est_signal)
        errors[i] = err
        print(f'for k={_k} error is {err}')

    plt.title(f'({rows}, {columns}), length={signal_length}, k={k}, std={noise_std}')
    plt.plot(ks, errors)
    plt.show()


if __name__ == '__main__':
    exp()
