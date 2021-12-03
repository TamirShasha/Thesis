import numpy as np
from scipy.signal import convolve
from skimage.draw import disk
import matplotlib.pyplot as plt
import os
import logging
from scipy.optimize import leastsq, least_squares

from src.utils.logger import logger
from src.algorithms.utils import log_size_S_2d_1axis, calc_mapping_2d, log_prob_all_is_noise, gram_schmidt
from src.utils.logsumexp import logsumexp_simple


class FilterEstimator2D:

    def __init__(self,
                 unnormalized_data: np.ndarray,
                 unnormalized_filter_basis: np.ndarray,
                 num_of_instances_range: (int, int),
                 prior_filter=None,
                 noise_std=1.,
                 noise_mean=0.,
                 estimate_noise_parameters=True,
                 signal_margin=0,
                 estimate_locations_and_num_of_instances=False,
                 experiment_dir=None,
                 experiment_attr=None,
                 plots=False,
                 log_level=logging.INFO):
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
        self.prior_filter = prior_filter
        self.estimate_noise_parameters = estimate_noise_parameters
        self.particle_margin = signal_margin
        self.estimate_locations_and_num_of_instances = estimate_locations_and_num_of_instances
        self.experiment_dir = experiment_dir
        self.experiment_attr = experiment_attr
        self.plots = plots

        logger.setLevel(log_level)

        if estimate_locations_and_num_of_instances:
            assert experiment_dir, 'Must provide experiment_dir for saving location'
            if experiment_attr is None:
                self.experiment_attr = {}

        if estimate_noise_parameters:
            noise_mean = np.nanmean(unnormalized_data)
            noise_std = np.nanstd(unnormalized_data)
            logger.info(f'Will estimate white noise parameters. '
                        f'Initial values are mean / std of entire data: '
                        f'({np.round(noise_mean, 3)} / {np.round(noise_std, 3)})')

        self.data = (unnormalized_data - noise_mean) / noise_std
        self.noise_std = 1  # We assume noise std is close to 1 after normalization
        self.data_size = self.data.shape[0]

        self.filter_basis_size = len(unnormalized_filter_basis)
        self.unmarginized_filter_basis, self.basis_norms = self.normalize_basis()
        self.signal_size = self.unmarginized_filter_basis[0].shape[0]

        # apply signal margin on filter basis
        self.filter_basis, self.signal_support = self.margin_filter_basis()
        self.summed_filters = np.nansum(self.filter_basis, axis=(1, 2))
        self.find_initial_filter_coeffs()

        # find max possible instances for given filter size
        self.max_possible_instances = min(
            self.num_of_instances_range[1],
            (self.data.shape[0] // self.signal_support) * (self.data.shape[1] // self.signal_support))
        logger.info(f'Maximum possible instances for size={self.signal_size} is {self.max_possible_instances}')
        self.min_possible_instances = self.num_of_instances_range[0]
        self.possible_instances = np.arange(self.min_possible_instances, self.max_possible_instances + 1)

        self.convolved_basis = self.convolve_basis()

        self.term_one = np.array([-self.calc_log_size_s(k) for k in self.possible_instances])
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

    def margin_filter_basis(self):
        """
        Adds margin around the filter to create artificial separation between signals
        :return:
        """
        filter_basis = np.array(
            [np.pad(x, ((self.particle_margin, self.particle_margin), (self.particle_margin, self.particle_margin)),
                    'constant', constant_values=((0, 0), (0, 0)))
             for x in self.unmarginized_filter_basis])
        signal_support = filter_basis[0].shape[0]
        return filter_basis, signal_support

    def find_initial_filter_coeffs(self):
        """
        If prior filter is given, finds optimal coeffs to match this filter
        If no prior filter was given, returns zero coeffs but first coeffs which is equal to noise_std
        """

        if self.prior_filter is None:
            initial_filter_params = np.zeros(self.filter_basis_size)
            initial_filter_params[0] = self.noise_std
            return initial_filter_params

        def to_minimize(params):
            return self.filter_basis.T.dot(params).flatten() - self.prior_filter

        initial_filter_params = least_squares(to_minimize, x0=np.zeros(self.filter_basis_size)).x

        logger.info(f'Optimal coeffs for prior filter are {initial_filter_params}')
        return initial_filter_params

    def calc_log_size_s(self, k):
        return log_size_S_2d_1axis(self.data.shape[0], k, self.signal_support)

    def calc_term_two(self, noise_mean):
        return log_prob_all_is_noise(self.data - noise_mean, self.noise_std)

    def calc_term_three_const(self):
        """
        The 3rd term of the likelihood function
        """
        return -np.arange(self.min_possible_instances, self.max_possible_instances + 1) / (2 * self.noise_std ** 2)

    def calc_term_four_const(self):
        """
        The 3rd term of the likelihood function
        """
        return -np.arange(self.min_possible_instances, self.max_possible_instances + 1) / (2 * self.noise_std ** 2)

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

    def calc_likelihood(self, mapping, filter_coeffs, noise_mean):
        """
        Calculates the likelihood of given model parameters
        """

        noise_term = log_prob_all_is_noise(self.data - noise_mean, 1)

        group_size_term = self.term_one
        filter_norm_const_term = -1 / (self.noise_std ** 2) * np.inner(filter_coeffs, filter_coeffs)
        noise_filter_const_term = -noise_mean / (self.noise_std ** 2) * np.inner(filter_coeffs, self.summed_filters)
        log_term = mapping[0, self.possible_instances]
        k_margin_term = group_size_term + \
                        self.possible_instances * (filter_norm_const_term + noise_filter_const_term) + \
                        log_term

        likelihood = noise_term + logsumexp_simple(k_margin_term)

        k_max = self.possible_instances[np.nanargmax(k_margin_term)]
        return likelihood, k_max

    def calc_gradient_discrete(self, filter_coeffs, noise_mean, likelihood):
        """
        calculate gradient with respect to each model parameter
        :param filter_coeffs: current point filter coefficients
        :param noise_mean: current estimated noise mean
        :param likelihood: current point likelihood
        :return: gradient of the filter coefficients (current point)
        """

        eps = 1e-4
        gradient = np.zeros(self.filter_basis_size + 1)  # last entry contains noise mean gradient

        for i in range(self.filter_basis_size):
            filter_coeffs_eps = filter_coeffs + np.eye(1, self.filter_basis_size, i)[0] * eps
            convolved_filter = self.calc_convolved_filter(filter_coeffs_eps)
            mapping = self.calc_mapping(convolved_filter)
            likelihood_eps, k_max = self.calc_likelihood(mapping, filter_coeffs_eps, noise_mean)
            gradient[i] = (likelihood_eps - likelihood) / eps

        return gradient

    def calc_likelihood_and_gradient(self, model_parameters):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """

        filter_coeffs, noise_mean = model_parameters[:-1], model_parameters[-1]

        convolved_filter = self.calc_convolved_filter(filter_coeffs)
        mapping = self.calc_mapping(convolved_filter)
        likelihood, k_max = self.calc_likelihood(mapping, filter_coeffs, noise_mean)
        gradient = self.calc_gradient_discrete(filter_coeffs, noise_mean, likelihood)

        return likelihood, gradient, k_max

    def optimize_parameters(self):

        curr_model_parameters, step_size, threshold, max_iter = np.zeros(self.filter_basis_size + 1), 0.1, 1e-4, 100
        curr_model_parameters[:self.filter_basis_size] = self.find_initial_filter_coeffs()

        curr_iter, diff = 0, np.inf
        curr_likelihood, curr_gradient, k_max = self.calc_likelihood_and_gradient(curr_model_parameters)
        while curr_iter < max_iter and diff > threshold:
            logger.debug(f'Current model parameters: {curr_model_parameters}, likelihood is {curr_likelihood}, '
                         f'k_max = {k_max}')
            next_model_parameters = curr_model_parameters + step_size * curr_gradient
            filter_coeffs = next_model_parameters[:-1]

            if self.estimate_noise_parameters:
                next_model_parameters[-1] = (np.nansum(self.data) -
                                             k_max * np.inner(filter_coeffs, self.summed_filters)) \
                                            / (self.data_size ** 2)

            next_likelihood, next_gradient, k_max = self.calc_likelihood_and_gradient(next_model_parameters)
            diff = np.abs(next_likelihood - curr_likelihood)

            step_size = np.abs(np.linalg.norm(next_model_parameters - curr_model_parameters)
                               / np.linalg.norm(next_gradient - curr_gradient))
            curr_model_parameters, curr_likelihood, curr_gradient = next_model_parameters, next_likelihood, next_gradient
            curr_iter += 1

        return curr_likelihood, curr_model_parameters

    def estimate(self):
        """
        Estimate optimal the match filter using given data samples and with respect to given filter basis
        :return: likelihood value and optimal unnormalized filter coefficient (can be used on user basis)
        """

        likelihood, optimal_model_parameters = self.optimize_parameters()

        optimal_filter_coeffs, optimal_noise_mean = optimal_model_parameters[:-1], optimal_model_parameters[-1]
        noise_std = np.nanstd(self.unnormalized_data)
        optimal_coeffs = optimal_filter_coeffs * noise_std / self.basis_norms

        if self.estimate_locations_and_num_of_instances:
            self.save_statistics(optimal_filter_coeffs, optimal_coeffs, optimal_noise_mean)

        return likelihood, optimal_coeffs

    def save_statistics(self, optimal_filter_coeffs, optimal_coeffs, optimal_noise_mean):
        convolved_filter = self.calc_convolved_filter(optimal_filter_coeffs)
        k = self.estimate_most_likely_num_of_instances(optimal_filter_coeffs, optimal_noise_mean, convolved_filter)
        logger.info(f'For size {self.signal_size} most likely k is {k}')
        self.find_optimal_signals_locations(convolved_filter, k)

        fig, axs = plt.subplots(nrows=1, ncols=2)

        matched_filter = self.filter_basis.T.dot(optimal_coeffs)
        pcm = axs[0].imshow(matched_filter, cmap='gray')
        axs[0].title.set_text('Matched Filter')
        plt.colorbar(pcm, ax=axs[0])

        center = matched_filter.shape[0] // 2
        filter_power_1d = np.square(matched_filter[center:-self.particle_margin, center])
        cum_filter_power_1d = np.nancumsum(filter_power_1d)
        relative_power_fraction = 1 - cum_filter_power_1d / cum_filter_power_1d[-1]
        xs = np.arange(0, filter_power_1d.shape[0]) * 2
        axs[1].plot(xs, relative_power_fraction)
        axs[1].title.set_text('Cumulative Power')
        fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_matched_filter.png')
        plt.savefig(fname=fig_path)
        plt.close()

    def estimate_most_likely_num_of_instances(self, filter_coeffs, noise_mean, convolved_filter):

        mapping = self.calc_mapping(convolved_filter)
        group_size_term = self.term_one
        filter_norm_const_term = -1 / (self.noise_std ** 2) * np.inner(filter_coeffs, filter_coeffs)
        noise_filter_const_term = -noise_mean / (self.noise_std ** 2) * np.inner(filter_coeffs, self.summed_filters)
        log_term = mapping[0, self.possible_instances]
        k_margin_term = group_size_term + \
                        self.possible_instances * (filter_norm_const_term + noise_filter_const_term) + \
                        log_term

        most_likely_num_of_instances = self.possible_instances[np.nanargmax(k_margin_term)]

        plt.title(f'Most likely number of instances for size {self.signal_size} is {most_likely_num_of_instances}')
        plt.plot(self.possible_instances, k_margin_term)
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
        if 'clean_data' not in self.experiment_attr:
            ax.imshow(self.data, cmap='gray')
        else:
            ax.imshow(self.experiment_attr['clean_data'], cmap='gray')

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

        return locations


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
