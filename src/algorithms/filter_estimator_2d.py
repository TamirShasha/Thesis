import numpy as np
from scipy.signal import convolve
from skimage.draw import disk
import matplotlib.pyplot as plt
import os

from src.utils.logger import logger
from src.algorithms.utils import log_size_S_2d_1axis, calc_mapping_2d, log_prob_all_is_noise, \
    _gradient_descent, gram_schmidt
from src.utils.logsumexp import logsumexp_simple


class FilterEstimator2D:

    def __init__(self,
                 unnormalized_data: np.ndarray,
                 unnormalized_filter_basis: np.ndarray,
                 num_of_instances_range: (int, int),
                 noise_std=1.,
                 noise_mean=0.,
                 signal_margin=0,
                 estimate_locations_and_num_of_instances=False,
                 experiment_dir=None,
                 plots=False,
                 logs=True):
        """
        initialize the filter estimator
        :param unnormalized_data: 1d data
        :param unnormalized_filter_basis: list of orthonormal basis
        """

        if len(unnormalized_filter_basis) == 0:
            raise Exception('Must provide basis')

        self.unnormalized_data = unnormalized_data
        self.unnormalized_filter_basis = unnormalized_filter_basis
        self.num_of_instances_range = num_of_instances_range
        self.noise_std = noise_std
        self.particle_margin = signal_margin
        self.estimate_locations_and_num_of_instances = estimate_locations_and_num_of_instances
        if estimate_locations_and_num_of_instances and experiment_dir is None:
            raise Exception('Must provide experiment_dir for saving location')
        self.experiment_dir = experiment_dir
        self.plots = plots

        self.data = (unnormalized_data - noise_mean) / noise_std
        self.filter_basis_size = len(unnormalized_filter_basis)
        self.unmarginized_filter_basis, self.basis_norms = self.normalize_basis()
        self.signal_size = self.unmarginized_filter_basis[0].shape[0]

        # apply signal margin on filter basis
        self.filter_basis = np.array(
            [np.pad(x, ((self.particle_margin, self.particle_margin), (self.particle_margin, self.particle_margin)),
                    'constant', constant_values=((0, 0), (0, 0)))
             for x in self.unmarginized_filter_basis])
        self.signal_support = self.filter_basis[0].shape[0]

        # find max possible instances for given filter size
        self.max_possible_instances = min(
            self.num_of_instances_range[1],
            (self.data.shape[0] // self.signal_support) * (self.data.shape[1] // self.signal_support))
        logger.info(f'Maximum possible instances for size={self.signal_size} is {self.max_possible_instances}')
        self.min_possible_instances = self.num_of_instances_range[0]

        self.convolved_basis = self.convolve_basis()

        self.term_one = -logsumexp_simple(
            np.array([self.calc_log_size_s(k) for k in
                      np.arange(self.min_possible_instances, self.max_possible_instances + 1)]))
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
        return log_size_S_2d_1axis(self.data.shape[0], k, self.signal_support)

    def calc_term_three_const(self):
        """
        The 3rd term of the likelihood function
        """
        return -np.arange(self.min_possible_instances, self.max_possible_instances + 1) / 2

    def convolve_basis_element(self, filter_element):
        """
        convolve one basis element with data
        :param filter_element: one element from filter basis
        :return: (n-d, m-d) while  n is data #rows, m is #columns
        """
        flipped_signal_filter = np.flip(filter_element)  # Flipping to cross-correlate
        conv = convolve(self.data, flipped_signal_filter, mode='valid')

        rings = np.zeros(shape=(self.signal_support, self.signal_support))
        for i in range(self.signal_support):
            for j in range(self.signal_support):
                rings[i, j] = int(np.linalg.norm([i - self.signal_support // 2, j - self.signal_support // 2]))

        convolution_size = self.data.shape[0] - self.signal_support
        radial_convolve = np.zeros(shape=(convolution_size, convolution_size))
        for row in np.arange(convolution_size):
            for col in np.arange(convolution_size):
                patch = np.copy(self.data[row: row + self.signal_support, col: col + self.signal_support])
                for r in np.arange(1, self.signal_support // 2 + 1):
                    patch[rings == r] = np.nanmean(patch[rings == r])
                radial_convolve[row, col] = np.nansum(np.inner(filter_element, patch))

        return conv

    def convolve_basis(self):
        """
        convolve each normalized basis element with data
        :return: (T, n-d, m-d) array where T is basis size, n is #rows, m is #columns
        """
        constants = np.zeros(shape=(self.filter_basis_size,
                                    self.data.shape[0] - self.signal_support + 1,
                                    self.data.shape[1] - self.signal_support + 1))
        for i, fil in enumerate(self.filter_basis):
            constants[i] = self.convolve_basis_element(fil)
        return constants

    def calc_convolved_filter(self, filter_coeffs):
        """
        :param filter_coeffs: coefficients for the filter basis
        :return: the dot product between relevant data sample and filter
        """
        convolved_filter = np.inner(self.convolved_basis.T, filter_coeffs).T
        return convolved_filter

    def calc_mapping(self, convolved_filter):
        """
        :param convolved_filter: dot product between data sample and a filter
        :return: likelihood mapping
        """
        return calc_mapping_2d(self.data.shape[0],
                               self.max_possible_instances,
                               self.signal_support,
                               convolved_filter)

    def calc_likelihood(self, filter_coeffs, mapping):
        term_three = logsumexp_simple(
            self.term_three_const * np.inner(filter_coeffs, filter_coeffs) + mapping[0, self.min_possible_instances:])
        likelihood = self.term_one + self.term_two + term_three
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
            gradient[i] = (likelihood_perturbation - likelihood) / eps

        return gradient

    def calc_likelihood_and_gradient(self, filter_coeffs):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """

        convolved_filter = self.calc_convolved_filter(filter_coeffs)
        mapping = self.calc_mapping(convolved_filter)
        likelihood = self.calc_likelihood(filter_coeffs, mapping)
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

        if self.estimate_locations_and_num_of_instances:
            convolved_filter = self.calc_convolved_filter(normalized_optimal_coeffs)
            k = self.estimate_most_likely_num_of_instances(optimal_coeffs, convolved_filter)
            logger.info(f'For size {self.signal_size} most likely k is {k}')
            self.find_optimal_signals_locations(convolved_filter, k)

            plt.imshow(self.filter_basis.T.dot(optimal_coeffs), cmap='gray')
            plt.colorbar()
            fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_matched_filter.png')
            plt.savefig(fname=fig_path)
            plt.close()

        return likelihood, optimal_coeffs

    def estimate_most_likely_num_of_instances(self, filter_coeffs, convolved_filter):
        mapping = self.calc_mapping(convolved_filter)
        term_two = self.term_three_const * np.inner(filter_coeffs, filter_coeffs)
        term_three = mapping[0, self.min_possible_instances:]
        likelihoods = self.term_one + term_two + term_three

        most_likely_num_of_instances = np.nanargmax(likelihoods) + self.min_possible_instances

        plt.title(f'Most likeliy number of instances for size {self.signal_size} is {most_likely_num_of_instances}')
        plt.plot(np.arange(self.min_possible_instances, self.max_possible_instances + 1), likelihoods)
        fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_instances_likelihoods.png')
        plt.savefig(fname=fig_path)
        plt.close()

        return most_likely_num_of_instances

    def find_optimal_signals_locations(self, convolved_filter, num_of_instances):

        n, m, k, d = self.data.shape[0], convolved_filter.shape[0], num_of_instances, self.signal_support
        max_k_in_row = min(n // d, k)

        # maximum likelihood for each row
        rows_best_loc_probs = np.full((m, n + 1, k + 1), -np.inf)
        rows_best_loc_probs[:, :, 0] = 0
        for row in range(m):
            for curr_k in range(1, max_k_in_row + 1):
                for i in range(n - curr_k * d, -1, -1):
                    curr_prob = convolved_filter[row, i] + rows_best_loc_probs[row, i + d, curr_k - 1]
                    prev_prob = rows_best_loc_probs[row, i + 1, curr_k]
                    if curr_prob > prev_prob:
                        rows_best_loc_probs[row, i, curr_k] = curr_prob
                    else:
                        rows_best_loc_probs[row, i, curr_k] = prev_prob

        rows_head_best_loc_probs = rows_best_loc_probs[:, 0, :]

        best_loc_probs = np.full((n, k + 1), -np.inf)
        best_loc_probs[m - 1] = rows_head_best_loc_probs[m - 1]
        best_loc_probs[:, 0] = 0
        best_ks = np.zeros(shape=(m, k + 1), dtype=int)
        for row in np.arange(m - 2, -1, -1):
            for curr_k in range(1, k + 1):
                probs = [rows_head_best_loc_probs[row, k_tag] + best_loc_probs[row + d, curr_k - k_tag]
                         for k_tag in range(1, curr_k + 1)]
                probs = [best_loc_probs[row + 1, curr_k]] + probs
                best_k = np.nanargmax(probs)
                best_ks[row, curr_k] = best_k
                best_loc_probs[row, curr_k] = probs[best_k]

        rows_and_number_of_instances = []
        left = k
        i = 0
        while i < m and left > 0:
            k_in_row = best_ks[i][left]
            if k_in_row > 0:
                rows_and_number_of_instances.append((i, k_in_row))
                left -= k_in_row
                i += d
            else:
                i += 1

        locations = []
        for (row, num) in rows_and_number_of_instances:
            pivot_idx = 0
            for j in np.arange(num, 0, -1):
                x = rows_best_loc_probs[row, pivot_idx:, j]
                idx = np.flatnonzero(np.diff(x[:n - num * d + 1]))[0]
                locations.append((pivot_idx + idx, row))
                pivot_idx += idx + d

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(self.data, cmap='gray')

        # Create a Rectangle patch
        for loc in locations:
            rect = patches.Rectangle(loc, d, d, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

        fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_locations.png')
        plt.title(f'Likely locations for size {self.signal_size}\n'
                  f'Likely number of instances is {k}')
        plt.savefig(fname=fig_path)
        plt.close()


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
