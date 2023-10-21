import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def _plot_1d(results):
    font = {'size': 12}
    import matplotlib
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 3))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[:2])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2 = fig.add_subplot(gs[2])

    ax1.plot(results['data'][:5000])
    ax1.plot(results['clean_data'][:5000])
    if results['noise_std'] == 0.1:
        ax1.set_ylim(-8, 8)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax2.yaxis.set_visible(False)
    ax2.plot(results['sizes_options'], results['likelihoods'], 'o-')
    ax2.axvline(x=results['signal_size'], label='signal true size', color='black', linestyle='--')
    ax2.scatter([results['most_likely_size']],
                [results['likelihoods'][results['most_likely_index']]],
                color='red', marker='*', s=300)

    plt.tight_layout()


def _plot_1d_full(results, shape='paraboly'):
    font = {'size': 12}
    import matplotlib
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2)

    measurement_ax = fig.add_subplot(gs[0, :2])
    measurement_ax.xaxis.set_visible(False)
    measurement_ax.yaxis.set_visible(False)
    measurement_ax.plot(results['data'][:5000])
    measurement_ax.plot(results['clean_data'][:5000])
    if results['noise_std'] == 0.1:
        measurement_ax.set_ylim(-8, 8)
    measurement_ax.autoscale(enable=True, axis='x', tight=True)

    likelihood_ax = fig.add_subplot(gs[1, 1])
    likelihood_ax.yaxis.set_visible(False)
    likelihood_ax.plot(results['sizes_options'], results['likelihoods'], 'o-')
    likelihood_ax.axvline(x=results['signal_size'], label='signal true size', color='black', linestyle='--')
    likelihood_ax.scatter([results['most_likely_size']],
                          [results['likelihoods'][results['most_likely_index']]],
                          color='red', marker='*', s=300, label='signal estimated size')
    likelihood_ax.legend()
    # likelihood_ax.autoscale(enable=True, axis='x', tight=True)

    signal_ax = fig.add_subplot(gs[1, 0])
    pad = results['signal_size'] // 10
    if shape == 'paraboly':
        from src.experiments.experiments_1d_new import paraboly
        signal_gen = paraboly
    else:
        from src.experiments.experiments_1d_new import arbitrary_signal
        signal_gen = arbitrary_signal

    signal_ax.plot(
        np.pad(signal_gen(results['signal_size']), (pad, pad), 'constant',
               constant_values=(0, 0)),
        label='true')
    signal_ax.plot(np.pad(results['estimated_signal'], (pad, pad), 'constant',
                          constant_values=(0, 0)), label='estimated', linestyle='--')
    signal_ax.legend(loc='upper right')

    plt.tight_layout()


def visualize_1d_experiment(exp_path, name, shape=None, save=False, plot=True):
    dir_path = r'C:\Users\tamir\Desktop\Plots For Paper\1d_experiments'

    data_path = os.path.join(exp_path, 'data.pkl')
    data = pickle.load(open(data_path, 'rb'))

    if shape is None:
        _plot_1d(data)
    else:
        _plot_1d_full(data, shape=shape)

    if save:
        fig_path = os.path.join(dir_path, f'{name}.png')
        plt.savefig(fname=fig_path)
    if plot:
        plt.show()

    plt.close()


def _plot_2d(results):
    print(results.keys())
    font = {'size': 14}
    import matplotlib
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(16, 4))
    mrc_fig = plt.subplot2grid((1, 4), (0, 0))
    likelihoods_fig = plt.subplot2grid((1, 4), (0, 2), colspan=2)
    # particle_fig = plt.subplot2grid((1, 5), (1, 0))
    # est_particle_fig = plt.subplot2grid((1, 5), (0, 2))
    clean_data_fig = plt.subplot2grid((1, 4), (0, 1))

    mrc_fig.imshow(results['data'], cmap='gray')
    mrc_fig.axis('off')

    # est_particle_fig.set_title('Estimated Signal')
    # pcm = est_particle_fig.imshow(results['estimated_signal'], cmap='gray')
    # est_particle_fig.axis('off')
    # plt.colorbar(pcm, ax=est_particle_fig)

    clean_data_fig.imshow(results['clean_data'], cmap='gray')
    clean_data_fig.axis('off')

    # likelihoods_fig.set_title('Likelihoods')
    likelihoods_fig.plot(results['sizes_options'], results['likelihoods'], 'o-')
    likelihoods_fig.axvline(x=results['signal_size'], label='signal true size', color='black', linestyle='--')
    likelihoods_fig.scatter([results['most_likely_size']],
                            [results['likelihoods'][results['most_likely_index']]],
                            color='red', marker='*', s=300, label='signal estimated size')
    # likelihoods_fig.axvline(x=self._results['most_likely_size'], label='signal estimated size', linestyle='--',
    #                         color='red')
    # likelihoods_fig.set_xlabel('Sizes')
    # likelihoods_fig.set_ylabel('Likelihood')
    likelihoods_fig.yaxis.set_visible(False)
    likelihoods_fig.legend()

    # if self._data_simulator:
    #     # particle_fig.set_title('True Signal')
    #     pcm = particle_fig.imshow(self._data_simulator.create_signal_instance(), cmap='gray')
    #     plt.colorbar(pcm, ax=particle_fig)

    plt.tight_layout()


def _plot_each_figure_2d(results):
    # font = {'size': 14}
    # import matplotlib
    # matplotlib.rc('font', **font)

    mrc_fig, mrc_ax = plt.subplots(1, figsize=(4, 4))
    mrc_ax.imshow(results['data'], cmap='gray')
    mrc_ax.axis('off')
    plt.tight_layout()
    plt.show()

    clean_data_fig, clean_data_ax = plt.subplots(1, figsize=(4, 4))
    clean_data_ax.imshow(results['clean_data'], cmap='gray')
    clean_data_ax.axis('off')
    plt.tight_layout()
    plt.show()

    est_particle_fig, est_particle_ax = plt.subplots(1, figsize=(4, 4))
    est_particle_ax.imshow(results['estimated_signal'], cmap='gray')
    # est_particle_ax.axis('off')
    plt.xticks(np.arange(0, 80, 10))
    plt.yticks(np.arange(0, 80, 10))
    # est_particle_fig.colorbar(ax=est_particle_ax)
    plt.tight_layout()
    plt.show()

    likelihoods_fig, likelihoods_ax = plt.subplots(1, figsize=(8, 4))
    likelihoods_ax.plot(results['sizes_options'], results['likelihoods'], 'o-')
    likelihoods_ax.axvline(x=results['signal_size'], label='signal true size', color='black', linestyle='--')
    likelihoods_ax.scatter([results['most_likely_size']],
                            [results['likelihoods'][results['most_likely_index']]],
                            color='red', marker='*', s=300, label='signal estimated size')
    likelihoods_ax.yaxis.set_visible(False)
    likelihoods_ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_2d_experiment(exp_path, name, shape=None, save=False, plot=True):
    dir_path = r'C:\Users\tamir\Desktop\Plots For Paper\2d_experiments'

    data_path = os.path.join(exp_path, 'data.pkl')
    data = pickle.load(open(data_path, 'rb'))
    data['clean_data'] = data['clean_data'][0]

    # _plot_2d(data)
    _plot_each_figure_2d(data)

    if save:
        fig_path = os.path.join(dir_path, f'{name}.png')
        plt.savefig(fname=fig_path)
    if plot:
        plt.show()

    plt.close()


if __name__ == '__main__':
    save = True

    # exp_1d_path = r'C:\Users\tamir\Desktop\Thesis\main\src\experiments\plots\03-07-2023\10-13-21_size_150_std_0.1'
    # visualize_1d_experiment(exp_1d_path, name='pulses_0_1', dir_path=dir_path, save=save)
    # exp_1d_path = r'C:\Users\tamir\Desktop\Thesis\main\src\experiments\plots\03-07-2023\10-21-37_size_150_std_3'
    # visualize_1d_experiment(exp_1d_path, name='pulses_3', dir_path=dir_path, save=save)
    # exp_1d_path = r'C:\Users\tamir\Desktop\Thesis\main\src\experiments\plots\03-07-2023\10-22-04_size_150_std_10'
    # visualize_1d_experiment(exp_1d_path, name='pulses_10', dir_path=dir_path, save=save)
    # exp_1d_path = r'C:\Users\tamir\Desktop\Thesis\main\src\experiments\plots\03-07-2023\10-22-34_size_150_std_50'
    # visualize_1d_experiment(exp_1d_path, name='pulses_50', dir_path=dir_path, save=save)

    # exp_1d_path_paraboly = r'C:\Users\tamir\Desktop\Thesis\main\src\experiments\plots\08-07-2023\17-31-27_size_100_std_1'
    # visualize_1d_experiment(exp_1d_path_paraboly, name='paraboly', shape='paraboly', dir_path=dir_path, save=save)

    # exp_1d_path_arbitrary = r'C:\Users\tamir\Desktop\Thesis\main\src\experiments\plots\08-07-2023\17-57-05_size_100_std_1'
    # visualize_1d_experiment(exp_1d_path_arbitrary, name='arbitrary', shape='arbitrary', dir_path=dir_path, save=save)

    exp_2d_disks_path = r'C:\Users\tamir\Desktop\Thesis\main\src\experiments\plots\11-07-2023\14-36-37_size_80_std_10'
    exp_2d_rings_path = r'C:\Users\tamir\Desktop\Thesis\main\src\experiments\plots\11-07-2023\15-50-42_size_80_std_10'
    visualize_2d_experiment(exp_2d_rings_path, name='disk', save=False)
