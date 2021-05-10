import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
from src.algorithms.length_extractor_1d import LengthExtractor1D
from src.experimental.length_extractor_1d_multiple_length import LengthExtractorML1D, CircleCutsDistribution, \
    Ellipse1t2CutsDistribution, SignalsDistribution
from src.algorithms.signal_power_estimator import estimate_signal_power, SignalPowerEstimator


class LengthExtractor2D:

    def __init__(self,
                 y,
                 length_options,
                 signal_filter_gen,
                 noise_mean,
                 noise_std,
                 signal_power_estimator_method,
                 exp_attr,
                 logs=True):
        self._y = y
        self._length_options = length_options
        self._signal_filter_gen = signal_filter_gen
        self._noise_mean = noise_mean
        self._noise_std = noise_std
        self._signal_power_estimator_method = signal_power_estimator_method
        self._logs = logs
        self._exp_attr = exp_attr
        self._n = self._y.shape[0]

        num_of_curves = 2 * int(np.log(np.max(self._y.shape)))
        self.to_delete = self._create_1d_data_from_curves(20)
        print(f'data length: {self.to_delete.shape[0]} x {self.to_delete.shape[1]}')

    def _rows_curves(self, jump=None):
        mat = self._y
        curve_rows = mat[np.arange(0, mat.shape[0], jump)]
        curve = np.concatenate(curve_rows)
        return curve

    def _random_lines_curves(self, num_of_curves):
        mat = self._y
        rows, columns = mat.shape

        curves = []
        for _ in range(num_of_curves):
            side = np.random.randint(2)
            if side == 0:  # up to down line
                up_column = np.random.randint(columns)
                down_column = np.random.randint(columns)
                rr, cc = line(0, up_column, rows - 1, down_column)
            else:  # left to right
                left_row = np.random.randint(rows)
                right_row = np.random.randint(rows)
                rr, cc = line(left_row, 0, right_row, columns - 1)

            curves.append(mat[rr, cc])

        return curves

    def _create_1d_data_from_curves(self, num_of_curves, strategy='random lines'):

        if strategy == 'rows':
            jump = self._y.shape[0] // num_of_curves
            return self._rows_curves(jump)

        if strategy == 'random lines':
            high_power_selection_factor = 3
            curves = self._random_lines_curves(high_power_selection_factor * num_of_curves)
            curves_powers = [
                estimate_signal_power(curve, self._noise_std, self._noise_mean, SignalPowerEstimator.FirstMoment)
                for curve in curves]
            top_curves = np.array(curves)[np.argsort(curves_powers)[-num_of_curves:]]
            return top_curves

    def _calc_for_distribution(self, signals_distributions, sep):
        num_of_curves = 2 * int(np.log(np.max(self._y.shape)))
        num_of_curves = 20

        times = len(self.to_delete)

        print(f'Will average over {times} runs, Num of curves is: {num_of_curves}')

        # data = np.concatenate(self._create_1d_data_from_curves(num_of_curves))
        data = self.to_delete

        best_ds = []
        sum_likelihoods = np.zeros_like(self._length_options)
        for t in range(times):
            print(f'At iteration {t}')
            likelihoods, d = LengthExtractorML1D(data=data[t],
                                                 length_distribution_options=signals_distributions,
                                                 noise_std=self._noise_std,
                                                 signal_separation=sep).extract()
            sum_likelihoods = sum_likelihoods + likelihoods
            curr_best_d = self._length_options[np.argmax(sum_likelihoods / times)]
            best_ds.append(curr_best_d)

        likelihoods = sum_likelihoods / times

        return likelihoods, best_ds

    def extract2(self):
        likelihoods = dict()
        best_ds = dict()

        # print(f'Running for circle distribution')
        # signals_distributions = [CircleCutsDistribution(length=l, filter_gen=self._signal_filter_gen)
        #                          for l in self._length_options]
        # circle_likelihoods, circle_best_ds = self._calc_for_distribution(signals_distributions, 0)
        # likelihoods['circle'] = circle_likelihoods
        # best_ds['circle'] = circle_best_ds

        # print(f'Running for circle distribution seperated')
        # signals_distributions = [CircleCutsDistribution(length=l, filter_gen=self._signal_filter_gen)
        #                          for l in self._length_options]
        # circle_likelihoods, circle_best_ds = self._calc_for_distribution(signals_distributions, 30)
        # likelihoods['circle_sep_30'] = circle_likelihoods
        # best_ds['circle_sep_30'] = circle_best_ds
        #
        # circle_likelihoods, circle_best_ds = self._calc_for_distribution(signals_distributions, 20)
        # likelihoods['circle_sep_20'] = circle_likelihoods
        # best_ds['circle_sep_20'] = circle_best_ds
        #
        # circle_likelihoods, circle_best_ds = self._calc_for_distribution(signals_distributions, 10)
        # likelihoods['circle_sep_10'] = circle_likelihoods
        # best_ds['circle_sep_10'] = circle_best_ds

        print(f'Running for ellipse 1:2 distribution')
        signals_distributions = [Ellipse1t2CutsDistribution(length=l, filter_gen=self._signal_filter_gen)
                                 for l in self._length_options]
        ellipse_likelihoods, ds = self._calc_for_distribution(signals_distributions, 0)
        likelihoods['ellipse'] = ellipse_likelihoods

        print(f'Running for ellipse 1:2 distribution')
        signals_distributions = [Ellipse1t2CutsDistribution(length=l, filter_gen=self._signal_filter_gen)
                                 for l in self._length_options]
        ellipse_likelihoods, ds = self._calc_for_distribution(signals_distributions, 30)
        likelihoods['ellipse_sep'] = ellipse_likelihoods

        # print('Running for 1d')
        # signals_distributions = [
        #     SignalsDistribution(length=l, cuts=[0, 0, 1], distribution=[0, 0, 1], filter_gen=self._signal_filter_gen)
        #     for l in self._length_options]
        # curr_likelihoods, dsp = self._calc_for_distribution(signals_distributions, 0)
        # likelihoods[f'1d'] = curr_likelihoods

        # for i in [7, 4]:
        #     cuts = [0.3, 0.5, 1]
        #     dist = [0, 0.1 * i, 1 - 0.1 * i]
        #     print(f'Running for {dist} distribution')
        #     signals_distributions = [
        #         SignalsDistribution(length=l, cuts=cuts, distribution=dist, filter_gen=self._signal_filter_gen)
        #         for l in self._length_options]
        #     curr_likelihoods = self._calc_for_distribution(signals_distributions, 0)
        #     likelihoods[f'dist_{dist}'] = curr_likelihoods

        # print('\nfor separation:\n')
        #
        # for cut in [0.5, 0.6]:
        #     for dist in [0.4, 0.5, 0.6]:
        #         print(f'Running for {dist} distribution')
        #         signals_distributions = [
        #             SignalsDistribution(length=l, cuts=[0, cut, 1], distribution=[0, dist, 1 - dist],
        #                                 filter_gen=self._signal_filter_gen)
        #             for l in self._length_options]
        #         curr_likelihoods = self._calc_for_distribution(signals_distributions, 30)
        #         likelihoods[f'{cut}_{np.round(dist, 1)}'] = curr_likelihoods

        # for i in [4, 3, 2, 0]:
        #     cuts = [0.3, 0.5, 1]
        #     dist = [0, 0.1 * i, 1 - 0.1 * i]
        #     print(f'Running for {dist} distribution')
        #     signals_distributions = [
        #         SignalsDistribution(length=l, cuts=cuts, distribution=dist, filter_gen=self._signal_filter_gen)
        #         for l in self._length_options]
        #     curr_likelihoods = self._calc_for_distribution(signals_distributions, 30)
        #     likelihoods[f'dist_{dist}_sep'] = curr_likelihoods

        #
        # class GeneralCutsDistribution(SignalsDistribution):
        #     def __init__(self, length: int, filter_gen):
        #         cuts = [0.4, 0.7, 1]
        #         distribution = [0.55, 0.35, 0.1]
        #         super().__init__(length, cuts, distribution, filter_gen)
        #
        # print(f'Running for general distribution')
        # signals_distributions = [GeneralCutsDistribution(length=l, filter_gen=self._signal_filter_gen)
        #                          for l in self._length_options]
        # general_likelihoods = self._calc_for_distribution(signals_distributions)
        # likelihoods['general1'] = general_likelihoods
        #
        # class GeneralCutsDistribution2(SignalsDistribution):
        #     def __init__(self, length: int, filter_gen):
        #         cuts = [0.2, 0.5, 0.8]
        #         distribution = [0.125, 0.325, 0.55]
        #         super().__init__(length, cuts, distribution, filter_gen)
        #
        # print(f'Running for general 2 distribution')
        # signals_distributions = [GeneralCutsDistribution2(length=l, filter_gen=self._signal_filter_gen)
        #                          for l in self._length_options]
        # general_likelihoods = self._calc_for_distribution(signals_distributions)
        # likelihoods['general2'] = general_likelihoods

        return likelihoods, best_ds

    def extract(self):
        # scan over the data to create 1d data
        num_of_curves = 2 * int(np.log(np.max(self._y.shape)))
        y_1d = self._create_1d_data_from_curves(num_of_curves)
        y_1d = self.to_delete

        print(f'data 1d length is {y_1d.shape[0]}')

        # create length extractor
        length_extractor_1d = LengthExtractor1D(y=y_1d,
                                                length_options=self._length_options,
                                                signal_filter_gen=self._signal_filter_gen,
                                                noise_mean=self._noise_mean,
                                                noise_std=self._noise_std,
                                                signal_power_estimator_method=self._signal_power_estimator_method,
                                                exp_attr=self._exp_attr,
                                                logs=self._logs)
        likelihoods, d = length_extractor_1d.extract()
        return likelihoods, d
