import time
import numpy as np
from scipy.signal import convolve
from skimage.draw import disk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import logging
from scipy.optimize import least_squares, minimize

from src.utils.logger import logger
from src.algorithms.utils import log_size_S_2d_1axis_multiple, \
    log_size_S_2d_1axis, \
    calc_mapping_2d, \
    log_prob_all_is_noise, \
    _calc_mapping_1d_many

from src.algorithms.utils import log_size_S_1d

from src.utils.logsumexp import logsumexp_simple


class FilterEstimator1D:

    def __init__(self,
                 unnormalized_data: np.ndarray,
                 unnormalized_filter_basis: np.ndarray,
                 num_of_instances_range: (int, int),
                 prior_filter=None,
                 noise_std_param=None,
                 noise_mean_param=None,
                 estimate_noise_parameters=True,
                 signal_margin=0,
                 save_statistics=False,
                 experiment_dir=None,
                 experiment_attr=None,
                 plots=False,
                 log_level=logging.INFO):
        """
        initialize the filter estimator
        """

        if len(unnormalized_filter_basis) == 0:
            raise Exception('Must provide basis')

        self.unnormalized_data = unnormalized_data
        self.unnormalized_filter_basis = unnormalized_filter_basis
        self.num_of_instances_range = num_of_instances_range
        self.prior_filter = prior_filter
        self.noise_mean_param = noise_mean_param
        self.noise_std_param = noise_std_param
        self.estimate_noise_parameters = estimate_noise_parameters
        self.particle_margin = signal_margin
        self.save_statistics = save_statistics
        self.experiment_dir = experiment_dir
        self.experiment_attr = experiment_attr
        self.plots = plots

        logger.setLevel(log_level)

        if save_statistics:
            assert experiment_dir, 'Must provide experiment_dir for saving location'
            if experiment_attr is None:
                self.experiment_attr = {}
            self.statistics = {
                "likelihoods": [],
                "noise_mean": []
            }

        # noise_mean = noise_mean_param if noise_mean_param is not None else np.nanmean(self.unnormalized_data)
        # noise_std = noise_std_param if noise_std_param is not None else np.nanstd(self.unnormalized_data)

        self.calculated_noise_means = np.array([
            noise_mean_param if noise_mean_param is not None else np.nanmean(data)
            for data in self.unnormalized_data])
        self.calculated_noise_stds = np.array([
            noise_std_param if noise_std_param is not None else np.nanstd(data)
            for data in self.unnormalized_data])

        self.data = (self.unnormalized_data - self.calculated_noise_means[:, np.newaxis]) / self.calculated_noise_stds[
                                                                                            :, np.newaxis]
        self.noise_mean = 0  # We assume noise mean is close to 0 after normalization
        self.noise_std = 1  # We assume noise std is close to 1 after normalization
        self.number_of_samples = self.data.shape[0]
        self.data_size = self.data.shape[1]

        self.filter_basis_size = len(self.unnormalized_filter_basis)
        self.unmarginized_filter_basis, self.basis_norms = self._normalize_basis()
        self.signal_size = self.unmarginized_filter_basis[0].shape[0]

        # apply signal margin on filter basis
        self.filter_basis, self.signal_support = self._margin_filter_basis()
        self.summed_filters = np.nansum(self.filter_basis, axis=1)

        # find max possible instances for given filter size
        self.max_possible_instances = min(self.num_of_instances_range[1], self.data_size // self.signal_support)
        logger.info(f'Maximum possible instances for size={self.signal_size} is {self.max_possible_instances}')
        self.min_possible_instances = self.num_of_instances_range[0]

        self.possible_instances = np.arange(self.min_possible_instances, self.max_possible_instances + 1)

        self.full_convolved_basis = None
        self.convolved_basis = None
        self.term_one = None
        self.term_three_const = None

        self.basic_row_col_jump = self.data_size // 200
        self.basic_row_col_jump = 1
        logger.info(f'Basic row col jump is {self.basic_row_col_jump}')

    def _normalize_basis(self):
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

    def _margin_filter_basis(self):
        """
        Adds margin around the filter to create artificial separation between signals
        :return:
        """
        filter_basis = np.array(
            [np.pad(x, (self.particle_margin, self.particle_margin), 'constant', constant_values=(0, 0))
             for x in self.unmarginized_filter_basis])
        signal_support = filter_basis[0].shape[0]
        return filter_basis, signal_support

    def _find_initial_filter_coeffs(self):
        """
        If prior filter is given, finds optimal coeffs to match this filter
        If no prior filter was given, returns zero coeffs but first coeffs which is equal to noise_std
        """

        if self.prior_filter is None:
            self.prior_filter = np.ones(self.signal_support)

        def to_minimize(params):
            return self.filter_basis.T.dot(params).flatten() - self.prior_filter.flatten()

        initial_filter_params = least_squares(to_minimize, x0=np.zeros(self.filter_basis_size)).x

        logger.info(f'Optimal coeffs for prior filter are {np.round(initial_filter_params, 3)}')
        return initial_filter_params

    def _calc_log_size_s(self):
        return log_size_S_1d(self.data_size, self.possible_instances[-1], self.signal_support)

    def _calc_term_two(self, noise_mean):
        return log_prob_all_is_noise(self.data - noise_mean, self.noise_std)

    def _calc_term_three_const(self):
        """
        The 3rd term of the likelihood function
        """
        return -np.arange(self.min_possible_instances, self.max_possible_instances + 1) / (2 * self.noise_std ** 2)

    def _calc_term_four_const(self):
        """
        The 3rd term of the likelihood function
        """
        return -np.arange(self.min_possible_instances, self.max_possible_instances + 1) / (2 * self.noise_std ** 2)

    def _convolve_basis_element(self, data_sample: np.ndarray, filter_element):
        """
        convolve one basis element with data
        :param data_sample: data sample
        :param filter_element: one element from filter basis
        :return: (n-d, m-d) while  n is data #rows, m is #columns
        """
        flipped_signal_filter = np.flip(filter_element)  # Flipping to cross-correlate
        conv = convolve(data_sample, flipped_signal_filter, mode='valid')
        return conv

    def _convolve_basis(self):
        """
        convolve each normalized basis element with data
        :return: (M, T, n-d, m-d) array where M is number of micrographs, T is basis size, n is #rows, m is #columns
        """
        constants = np.zeros(shape=(self.number_of_samples,
                                    self.filter_basis_size,
                                    self.data_size - self.signal_support + 1))
        for i, data_sample in enumerate(self.data):
            for j, fil in enumerate(self.filter_basis):
                constants[i][j] = self._convolve_basis_element(data_sample, fil)
        return constants

    def _calc_convolved_filter(self, filter_coeffs, full=False):
        """
        :param filter_coeffs: coefficients for the filter basis
        :return: the dot product between relevant data sample and filter
        """
        if full:
            convolved_filter = np.inner(self.full_convolved_basis.transpose([0, 2, 1]), filter_coeffs)
        else:
            convolved_filter = np.inner(self.convolved_basis.transpose([0, 2, 1]), filter_coeffs)
        return convolved_filter

    def _calc_mapping(self, convolved_filter: np.ndarray, axis=0):
        """
        :param convolved_filter: dot product between data sample and a filter
        :param if axis=0, will calculate very well separation over rows, if axis=1 will calculate over columns
        :return: likelihood mapping
        """

        _data_size = np.ceil(self.data_size / self.basic_row_col_jump).astype(int)
        _signal_support = np.ceil(self.signal_support / self.basic_row_col_jump).astype(int)
        # _row_jump = np.ceil(self.row_jump / self.basic_row_col_jump).astype(int)
        # _convolved_filter = convolved_filter[::self.basic_row_col_jump, ::self.basic_row_col_jump]

        mapping = calc_mapping_2d(convolved_filter.shape[0] + _signal_support - 1,
                                  self.max_possible_instances,
                                  _signal_support,
                                  convolved_filter if axis == 0 else convolved_filter.T,
                                  row_jump=_row_jump)

        return mapping

    def _calc_likelihood(self, filter_coeffs, noise_mean):
        """
        Calculates the likelihood of given model parameters
        """
        start_time = time.time()
        convolved_filter = self._calc_convolved_filter(filter_coeffs)
        log_terms = np.array(
            [self._calc_mapping(convolved_filter[i], axis=0)
             for i in range(self.number_of_samples)])[:, 0, self.possible_instances]  # vws on rows
        # log_term_columns = self._calc_mapping(convolved_filter, axis=1)[0, self.possible_instances]  # vws on columns

        noise_terms = np.array([
            log_prob_all_is_noise(self.data[i] - noise_mean, self.noise_std) for i in range(self.number_of_samples)
        ])

        # noise_terms = np.array([0 for _ in range(self.number_of_micrographs)])

        filter_norm_const_term = -1 / (2 * self.noise_std ** 2) * np.inner(filter_coeffs, filter_coeffs)
        noise_filter_const_term = -noise_mean / (self.noise_std ** 2) * np.inner(filter_coeffs, self.summed_filters)
        # log_term = np.logaddexp(log_term_rows, log_term_columns)

        if len(self.possible_instances) > 1:
            k_margin_term = self.term_one + \
                            self.possible_instances * (filter_norm_const_term + noise_filter_const_term) + \
                            log_terms

            likelihood = noise_terms + logsumexp_simple(k_margin_term)

            # Calculate number of instances expectation
            k_probabilities = np.exp(k_margin_term - logsumexp_simple(k_margin_term))
            k_expectation = np.nansum(self.possible_instances * k_probabilities)

            return_val = np.nanmean(likelihood, axis=0), k_expectation
        else:
            k = self.possible_instances[0]
            likelihood = self.term_one + noise_terms + k * (
                    filter_norm_const_term + noise_filter_const_term) + log_terms[:, 0]
            return_val = np.nanmean(likelihood, axis=0), k

        end_time = time.time()
        # logger.info(f'Took {end_time - start_time} seconds, likelihood {return_val}')
        return return_val

    def _calc_likelihood_for_locations(self, filter_coeffs, convolved_filter, noise_mean, locations):
        """
        Calculates the likelihood for given locations
        """

        noise_terms = - 1 / (self.noise_std ** 2) * np.linalg.norm(self.data, axis=(1, 2))
        convolve_term = np.array(
            [np.nansum(convolved_filter[i][np.array(locations[i])[:, 1], np.array(locations[i])[:, 0]])
             for i in range(self.number_of_samples)])

        filter_norm_const_term = -1 / (2 * self.noise_std ** 2) * np.inner(filter_coeffs, filter_coeffs)
        noise_filter_const_term = -noise_mean / (self.noise_std ** 2) * np.inner(filter_coeffs, self.summed_filters)
        ks = np.array([len(locs) for locs in locations])

        likelihoods = noise_terms + ks * (filter_norm_const_term + noise_filter_const_term) + convolve_term

        return np.nanmean(likelihoods, axis=0), ks

    def _calc_gradient_discrete(self, filter_coeffs, noise_mean, likelihood):
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
            likelihood_eps, k_expectation = self._calc_likelihood(filter_coeffs_eps, noise_mean)
            gradient[i] = (likelihood_eps - likelihood) / eps

        return gradient

    def _calc_likelihood_and_gradient(self, model_parameters):
        """
        Calculates the likelihood function for given filter coefficients and its gradient
        """

        filter_coeffs, noise_mean = model_parameters[:-1], model_parameters[-1]

        likelihood, k_expectation = self._calc_likelihood(filter_coeffs, noise_mean)
        gradient = self._calc_gradient_discrete(filter_coeffs, noise_mean, likelihood)

        return likelihood, gradient, k_expectation

    def _optimize_params(self):
        start_time = time.time()

        filter_coeffs = self._find_initial_filter_coeffs()
        noise_mean = self.noise_mean

        alternate_iteration = 5
        max_alternate_iterations = 10
        tol = 1

        def func(_filter_coeffs):
            _likelihood, _k_expectation = self._calc_likelihood(_filter_coeffs, noise_mean)
            logger.debug(
                f'Current model parameters: {np.round(_filter_coeffs, 3)}, noise mean: {np.round(noise_mean, 3)}, '
                f'likelihood is {np.round(_likelihood, 4)}, '
                f'k expectation = {_k_expectation}')
            return -_likelihood

        if not self.estimate_noise_parameters:
            logger.debug(
                f'Optimizing model parameters for maximum of {max_alternate_iterations * alternate_iteration} iterations '
                f'and tolerance of {tol}.')

            result = minimize(func, filter_coeffs,
                              tol=1e-2,
                              method='BFGS',
                              options={
                                  'disp': False,
                                  'maxiter': alternate_iteration * max_alternate_iterations
                              })
        else:
            logger.debug(
                f'Optimizing model parameters for maximum of {max_alternate_iterations}x{alternate_iteration} iterations '
                f'and tolerance of {tol}.')
            for i in range(max_alternate_iterations):
                # Update filter coefficients
                result = minimize(func, filter_coeffs,
                                  tol=1e-2,
                                  method='BFGS',
                                  options={
                                      'disp': False,
                                      'maxiter': alternate_iteration
                                  })

                filter_coeffs = result.x

                kernel_size, percentile = 30, 3
                convolved_filter = self._calc_convolved_filter(filter_coeffs)
                convolved_patch = convolve(convolved_filter, np.ones((kernel_size, kernel_size)) / (kernel_size ** 2),
                                           mode='valid')
                perc = np.percentile(convolved_patch.ravel(), percentile)
                idxs = np.argwhere(convolved_patch < perc)

                max_iters = 20
                marks = np.zeros_like(self.data)
                chosen_idxs = []
                chosen_idx = idxs[np.random.choice(np.arange(len(idxs)), 1)][0]
                marks[chosen_idx[0]: chosen_idx[0] + 30, chosen_idx[1]: chosen_idx[1] + 30] = 1
                chosen_idxs.append(chosen_idx)
                for _ in range(10):
                    chosen_idx = idxs[np.random.choice(np.arange(len(idxs)), 1)][0]
                    it = 0
                    while np.any(
                            marks[chosen_idx[0]: chosen_idx[0] + 30,
                            chosen_idx[1]: chosen_idx[1] + 30]) and it < max_iters:
                        chosen_idx = idxs[np.random.choice(np.arange(len(idxs)), 1)][0]
                        it += 1
                    marks[chosen_idx[0]: chosen_idx[0] + 30, chosen_idx[1]: chosen_idx[1] + 30] = 1
                    chosen_idxs.append(chosen_idx)

                noise_mean = np.nanmedian(self.data[marks.astype(bool)])

                if self.save_statistics:
                    fig, axs = plt.subplots(2, 2)
                    axs[0, 0].set_title('Raw data')
                    axs[0, 0].imshow(self.data, cmap='gray')
                    axs[0, 1].set_title('Convolved Filter')
                    axs[0, 1].imshow(convolved_filter, cmap='gray')
                    axs[1, 0].set_title('Convolved Patch')
                    axs[1, 0].imshow(convolved_patch, cmap='gray')
                    axs[1, 1].set_title(f'Under percentile {percentile}')
                    axs[1, 1].imshow(convolved_patch < perc)

                    for idx in chosen_idxs:
                        axs[1, 1].scatter(idx[1], idx[0])

                    for idx in chosen_idxs:
                        rect = patches.Rectangle(np.flip(idx) + 30, 30, 30, color='red')
                        axs[0, 0].add_patch(rect)
                        # rect = patches.Rectangle(np.flip(idx)+30, 30, 30)
                        # axs[0, 1].add_patch(rect)
                    # axs[0].scatter(chosen_idx[:, 0], chosen_idx[:, 1])
                    fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_noise_locations.png')
                    plt.savefig(fname=fig_path)
                    plt.close()

                if result.success:
                    logger.info('SUCCESS. Optimizing model parameters reached its tolerance.')
                    break
                elif i == max_alternate_iterations:
                    logger.info('Failed to reach tolerance while optimizing model parameters.')

        model_parameters = np.concatenate([result.x, [noise_mean]])

        logger.info(f'Optimized model parameters: {np.round(model_parameters, 3)}, '
                    f'Total Time: {np.round(time.time() - start_time)} seconds')

        return -result.fun, model_parameters

    def _optimize_parameters(self):

        curr_model_parameters, step_size, threshold, max_iter = np.zeros(self.filter_basis_size + 1), 0.1, 1e-3, 100
        curr_model_parameters[:self.filter_basis_size] = self._find_initial_filter_coeffs()

        curr_iter, diff = 0, np.inf
        curr_likelihood, curr_gradient, k_expectation = self._calc_likelihood_and_gradient(curr_model_parameters)

        update_filter_coeffs = True
        stall_noise_mean_update = 3  # iterations
        while curr_iter < max_iter and diff > threshold:
            iter_start_time = time.time()

            if self.save_statistics:
                self.statistics['likelihoods'].append(curr_likelihood)
                self.statistics['noise_mean'].append(curr_model_parameters[-1])

            logger.debug(
                f'(#{curr_iter + 1}) '
                f'Current model parameters: {np.round(curr_model_parameters, 3)}, '
                f'likelihood is {np.round(curr_likelihood, 4)}, '
                f'k expectation = {k_expectation}')

            if update_filter_coeffs:
                logger.debug(f'Updates filter coefficients')
                next_model_parameters = curr_model_parameters + step_size * curr_gradient
                if self.estimate_noise_parameters:
                    if stall_noise_mean_update == 0:
                        update_filter_coeffs = not update_filter_coeffs
                    else:
                        logger.debug(f'# iterations until noise update: {stall_noise_mean_update}')
                        stall_noise_mean_update -= 1
            else:
                logger.debug(f'Updates noise mean')
                filter_coeffs = curr_model_parameters[:-1]
                next_model_parameters = curr_model_parameters.copy()
                next_model_parameters[-1] = (np.nansum(self.data) -
                                             k_expectation * np.inner(filter_coeffs, self.summed_filters)) \
                                            / (self.data_size ** 2)
                update_filter_coeffs = not update_filter_coeffs

            next_likelihood, next_gradient, k_expectation = self._calc_likelihood_and_gradient(next_model_parameters)
            diff = np.abs(next_likelihood - curr_likelihood)

            step_size = np.abs(np.linalg.norm(next_model_parameters - curr_model_parameters)
                               / np.linalg.norm(next_gradient - curr_gradient))
            curr_model_parameters, curr_likelihood, curr_gradient = next_model_parameters, next_likelihood, next_gradient
            curr_iter += 1

            iter_total_time = time.time() - iter_start_time

        logger.info(
            f'(FINISH) '
            f'Current model parameters: {np.round(curr_model_parameters, 3)}, '
            f'likelihood is {np.round(curr_likelihood, 4)}, '
            f'k expectation: {k_expectation}, '
            f'Total Iteration time: {iter_total_time}')

        return curr_likelihood, curr_model_parameters

    def _generate_and_save_statistics(self, optimal_filter_coeffs, optimal_coeffs, optimal_noise_mean):

        # # Save evolving likelihood graph
        # plt.title('Likelihood per optimization (GD) iteration')
        # plt.plot(self.statistics['likelihoods'][len(self.statistics['likelihoods']) // 2:])
        # fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_gd_likelihoods.png')
        # plt.savefig(fname=fig_path)
        # plt.close()
        #
        # # Save evolving noise mean graph
        # plt.title('Noise Mean per optimization (GD) iteration')
        # plt.plot(self.statistics['noise_mean'])
        # fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_gd_noise_mean.png')
        # plt.savefig(fname=fig_path)
        # plt.close()
        #
        # # Save num of instances likelihoods graph as fig
        convolved_filter = self._calc_convolved_filter(optimal_filter_coeffs)

        for i, mrc in enumerate(self.data):
            k = self._estimate_most_likely_num_of_instances(optimal_filter_coeffs, optimal_noise_mean,
                                                            convolved_filter[i])
            logger.info(f'MRC#{i + 1} For size {self.signal_size} most likely k is {k}')

        # Save Smax locations as fig
        mrc_locations = self._find_optimal_signals_locations(convolved_filter, k)
        likelihood_of_locations, ks = self._calc_likelihood_for_locations(optimal_filter_coeffs,
                                                                          convolved_filter,
                                                                          optimal_noise_mean,
                                                                          mrc_locations // self.basic_row_col_jump)

        particles_patches = []
        for mrc_idx, mrc in enumerate(self.data):
            locations = mrc_locations[mrc_idx]

            intersection_img = np.zeros_like(self.data[0])
            for _loc in locations:
                loc = np.flip(_loc + self.particle_margin)
                particles_patches.append(mrc[loc[0]: loc[0] + self.signal_size, loc[1]: loc[1] + self.signal_size])
                intersection_img[loc[0]: loc[0] + self.signal_size, loc[1]: loc[1] + self.signal_size] += 1

            # plt.imshow(intersection_img)
            # plt.colorbar()
            # plt.show()

        particles_patches = np.array(particles_patches)
        normed_particles_patches = \
            particles_patches / np.linalg.norm(particles_patches, axis=(1, 2))[:, np.newaxis, np.newaxis]

        correlations = []
        for i in range(normed_particles_patches.shape[0]):
            for j in range(i):
                correlations.append(np.dot(normed_particles_patches[i].ravel(),
                                           normed_particles_patches[j].ravel()))
        correlations = np.array(correlations)

        plt.title(f'Correlation Median: {np.median(correlations)}')
        plt.hist(correlations, bins=20)
        fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_correlation_hist.png')
        plt.savefig(fname=fig_path)
        plt.close()
        logger.debug(f'({self.signal_size}) Correlation median: {np.median(correlations)}')

        average_patch = np.nanmean(normed_particles_patches, axis=0)
        plt.title('Average patch of maximum probability locations\n')
        plt.imshow(average_patch)
        plt.colorbar()
        fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_average_patch.png')
        plt.savefig(fname=fig_path)
        plt.close()

        # power_distribution_score = np.linalg.norm(average_patch * (self.filter_basis[0] != 0)) / np.linalg.norm(
        #     average_patch)
        inner_patch = average_patch * (self.filter_basis[0] != 0)
        stats_inner = np.nanmean(inner_patch[inner_patch != 0]), np.nanstd(inner_patch[inner_patch != 0])
        outer_patch = average_patch * (self.filter_basis[0] == 0)
        stats_outer = np.nanmean(outer_patch[outer_patch != 0]), np.nanstd(outer_patch[outer_patch != 0])
        power_distribution_score = (stats_inner[0] / stats_outer[0], stats_inner[1] / stats_outer[1])

        logger.debug(f'({self.signal_size}) Power distribution score: {power_distribution_score}')

        fig, axs = plt.subplots(nrows=1, ncols=2)

        # Save matched filter and cumulative power as fig
        matched_filter = self.filter_basis.T.dot(optimal_coeffs[0])
        pcm = axs[0].imshow(matched_filter, cmap='gray')
        axs[0].title.set_text('Matched Filter')
        plt.colorbar(pcm, ax=axs[0])

        center = matched_filter.shape[0] // 2
        if self.particle_margin > 0:
            filter_power_1d = np.square(matched_filter[center:-self.particle_margin, center])
        else:
            filter_power_1d = np.square(matched_filter[center:, center])
        cum_filter_power_1d = np.nancumsum(filter_power_1d)
        relative_power_fraction = 1 - cum_filter_power_1d / cum_filter_power_1d[-1]
        xs = np.arange(0, filter_power_1d.shape[0]) * 2
        axs[1].plot(xs, relative_power_fraction)
        axs[1].title.set_text('Cumulative Power')
        fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_matched_filter.png')
        plt.savefig(fname=fig_path)
        plt.close()

        # Return some of the statistics
        return correlations, power_distribution_score, likelihood_of_locations

    def _estimate_most_likely_num_of_instances(self, filter_coeffs, noise_mean, convolved_filter):

        mapping = self._calc_mapping(convolved_filter)
        group_size_term = self.term_one
        filter_norm_const_term = -1 / (self.noise_std ** 2) * np.inner(filter_coeffs, filter_coeffs)
        noise_filter_const_term = -noise_mean / (self.noise_std ** 2) * np.inner(filter_coeffs, self.summed_filters)
        log_term = mapping[0, self.possible_instances]
        k_margin_term = group_size_term + \
                        self.possible_instances * (filter_norm_const_term + noise_filter_const_term) + \
                        log_term

        most_likely_num_of_instances = self.possible_instances[np.nanargmax(k_margin_term)]

        # plt.title(f'Most likely number of instances for size {self.signal_size} is {most_likely_num_of_instances}')
        # plt.plot(self.possible_instances, k_margin_term)
        # fig_path = os.path.join(self.experiment_dir, f'{self.signal_size}_instances_likelihoods.png')
        # plt.savefig(fname=fig_path)
        # plt.close()

        return most_likely_num_of_instances

    def _find_optimal_signals_locations(self, convolved_filters, num_of_instances) -> np.ndarray:

        mrc_locations = []

        for mrc_idx, mrc in enumerate(self.data):

            convolved_filter = convolved_filters[mrc_idx]

            _signal_support = np.ceil(self.signal_support / self.basic_row_col_jump).astype(int)
            _data_size = convolved_filter.shape[0] + _signal_support - 1
            n, m, k, d = _data_size, convolved_filter.shape[0], num_of_instances, _signal_support
            row_jump = np.ceil(self.row_jump / self.basic_row_col_jump).astype(int)
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
                    probs = [rows_head_best_loc_probs[row, k_tag] + best_loc_probs[row + row_jump, curr_k - k_tag]
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
                    i += row_jump
                else:
                    i += 1

            locations = []
            for (row, num) in rows_and_number_of_instances:
                pivot_idx = 0
                for j in np.arange(num, 0, -1):
                    x = np.copy(rows_best_loc_probs[row, pivot_idx:, j][:n - num * d + 1])
                    x[x == -np.inf] = 0
                    diffs = np.diff(x)
                    if np.all(diffs == 0):
                        idx = len(diffs)
                    else:
                        idx = np.flatnonzero(diffs)[0]
                    locations.append((pivot_idx + idx, row))
                    pivot_idx += idx + d

            locations = np.array(locations)
            locations *= self.basic_row_col_jump

            # Create figure and axes
            fig, ax = plt.subplots()

            # Display the image
            if 'clean_data' not in self.experiment_attr:
                ax.imshow(mrc, cmap='gray')
            else:
                ax.imshow(self.experiment_attr['clean_data'][mrc_idx], cmap='gray')

            # Create a Rectangle patch
            for loc in locations:
                rect = patches.Rectangle(loc + self.particle_margin, self.signal_size, self.signal_size,
                                         linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)

            fig_path = os.path.join(self.experiment_dir, f'mrc#{mrc_idx + 1}_{self.signal_size}_locations.png')
            plt.title(f'Likely locations for size {self.signal_size}\n'
                      f'Likely number of instances is {k}')
            plt.savefig(fname=fig_path)
            plt.close()

            mrc_locations.append(locations)

        return np.array(mrc_locations)

    def estimate(self):
        """
        Estimate optimal the match filter using given data samples and with respect to given filter basis
        :return: likelihood value and optimal unnormalized filter coefficient (can be used on user basis)
        """

        if self.min_possible_instances > self.max_possible_instances:
            logger.info(f'Minimum possible instances is larger than maximum possible instances, '
                        f'will set likelihood to -np.inf')
            return -np.inf, np.zeros_like(self.filter_basis_size)

        self.full_convolved_basis = self._convolve_basis()
        self.convolved_basis = self.full_convolved_basis[:, :, ::self.basic_row_col_jump]

        self.term_one = -self._calc_log_size_s()
        self.term_three_const = self._calc_term_three_const()

        likelihood, optimal_model_parameters = self._optimize_params()
        # likelihood, optimal_model_parameters = self._optimize_parameters()

        optimal_filter_coeffs, optimal_noise_mean = optimal_model_parameters[:-1], optimal_model_parameters[-1]
        self._calc_likelihood(optimal_filter_coeffs, 0)
        optimal_coeffs = np.array(
            [optimal_filter_coeffs * noise_std / self.basis_norms for noise_std in self.calculated_noise_stds])
        # optimal_coeffs = optimal_filter_coeffs * noise_std / self.basis_norms

        if self.save_statistics:
            correlations, power_distribution_score, likelihood_of_locations = \
                self._generate_and_save_statistics(optimal_filter_coeffs, optimal_coeffs, optimal_noise_mean)

            return likelihood, optimal_coeffs[0], correlations, power_distribution_score, likelihood_of_locations
        else:
            return likelihood, optimal_coeffs[0]
