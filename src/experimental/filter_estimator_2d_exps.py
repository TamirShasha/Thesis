import matplotlib.pyplot as plt
import numpy as np

from src.experiments.data_simulator_2d import DataSimulator2D, Shapes2D
from src.algorithms.filter_estimator_2d import FilterEstimator2D, create_filter_basis


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
    signal_fraction = 1 / 7
    # signal_gen = lambda l, p: Shapes2D.double_disk(l, l // 2, -p // 2, p)
    signal_gen = Shapes2D.sphere
    noise_std = 0.01
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
    print(k)

    # plt.imshow(signal)
    # plt.colorbar()
    # plt.show()
    plt.imshow(data, cmap='gray')
    plt.show()

    filter_basis = create_filter_basis(signal_length, 5, basis_type='rings')
    filter_estimator = FilterEstimator2D(data, filter_basis, k // 2, noise_std)
    likelihood, optimal_coeffs = filter_estimator.estimate()
    est_signal = filter_basis.T.dot(optimal_coeffs)

    print(optimal_coeffs)
    err = calc_error(signal, est_signal)
    print(f'err for normal basis {err}')
    print(f'likelihood for normal basis {likelihood}')
    plt.title(f'({rows}, {columns}), length={signal_length}, k={k}, std={noise_std}, \nerror={np.round(err, 3)}')

    filter_basis_cheb = create_filter_basis(signal_length, 5)
    # filter_basis_cheb = create_basis(signal_length, 5)
    filter_estimator = FilterEstimator2D(data, filter_basis_cheb, k // 2, noise_std)
    likelihood_cheb, optimal_coeffs_cheb = filter_estimator.estimate()
    est_signal_cheb = filter_basis_cheb.T.dot(optimal_coeffs_cheb)
    err = calc_error(signal, est_signal_cheb)
    print(f'err for cheb basis {err}')
    print(f'likelihood for cheb basis {likelihood_cheb}')

    filter_basis_signal = np.array([signal])
    filter_estimator = FilterEstimator2D(data, filter_basis_signal, k // 2, noise_std)
    likelihood_signal, optimal_coeffs_signal = filter_estimator.estimate()
    est_signal_signal = filter_basis_signal.T.dot(optimal_coeffs_signal)
    err = calc_error(signal, est_signal_signal)
    print(optimal_coeffs_signal)
    print(f'err for signal basis {err}')
    print(f'likelihood for signal basis {likelihood_signal}')

    plt.imshow(signal)
    plt.show()
    plt.imshow(est_signal)
    plt.colorbar()
    plt.show()

    plt.imshow(est_signal_cheb)
    plt.colorbar()
    plt.show()

    # plot_signal_3d(est_signal)


def exp2():
    rows = 1200
    columns = 1200
    signal_length = 80
    signal_power = 1
    signal_fraction = 1 / 8
    # signal_gen = lambda l, p: Shapes2D.double_disk(l, l // 2, -p // 2, p)
    signal_gen = Shapes2D.sphere
    noise_std = .1
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
    print(k)

    # plt.imshow(signal)
    # plt.colorbar()
    # plt.show()
    plt.imshow(data, cmap='gray')
    plt.show()

    filter_basis = create_filter_basis(signal_length, 5)

    ks = [1, 5, 10, 15, 20, 25, 30, 35, 45, 60, 80]
    # ks = [k//10, k//8, k//5, k//3, k//2, k, int(k*1.2), int(k*1.5), int(k*2)]
    print(ks)
    # ks = [10]
    errors = np.zeros_like(ks, dtype=float)
    likelihoods = np.zeros_like(ks, dtype=float)
    for i, _k in enumerate(ks):
        filter_estimator = FilterEstimator2D(data, filter_basis, _k)
        likelihoods[i], optimal_coeffs = filter_estimator.estimate()
        est_signal = filter_basis.T.dot(optimal_coeffs)
        err = calc_error(signal, est_signal)
        errors[i] = err
        print(f'for k={_k} error is {err}')

    plt.title(f'({rows}, {columns}), length={signal_length}, k={k}, std={noise_std}')
    plt.plot(ks, likelihoods)
    plt.show()

    plt.title(f'({rows}, {columns}), length={signal_length}, k={k}, std={noise_std}')
    plt.plot(ks, errors)
    plt.show()


def exp3():
    rows = 1500
    columns = 1500
    signal_length = 100
    signal_power = 1
    signal_fraction = 1 / 5
    # signal_gen = lambda l, p: Shapes2D.double_disk(l, l // 2, -p // 2, p)
    signal_gen = Shapes2D.sphere
    noise_std = .1
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
                               apply_ctf=apply_ctf,
                               method='VWS')

    data = sim_data.simulate()
    signal = sim_data.create_signal_instance()
    k = sim_data.occurrences
    print(k)

    # plt.imshow(signal)
    # plt.colorbar()
    # plt.show()
    plt.imshow(data, cmap='gray')
    plt.show()

    filter_basis = create_filter_basis(signal_length, 20)

    # ks = [1, 5, 10, 15, 20, 25, 30, 35, 45, 60, 80]
    # ks = np.concatenate([[1, 5], np.arange(10, 121, 10)])
    ks = np.arange(40, 71, 10)
    # ks = [k//10, k//8, k//5, k//3, k//2, k, int(k*1.2), int(k*1.5), int(k*2)]
    print(ks)
    # ks = [10]
    errors = np.zeros_like(ks, dtype=float)
    likelihoods = np.zeros_like(ks, dtype=float)
    for i, _k in enumerate(ks):
        filter_estimator = FilterEstimator2D(data, filter_basis, _k)
        likelihoods[i], optimal_coeffs = filter_estimator.estimate()
        est_signal = filter_basis.T.dot(optimal_coeffs)
        err = calc_error(signal, est_signal)
        errors[i] = err
        print(f'for k={_k} error is {err}')

    plt.title(f'({rows}, {columns}), length={signal_length}, k={k}, std={noise_std}')
    plt.plot(ks, likelihoods)
    plt.show()

    plt.title(f'({rows}, {columns}), length={signal_length}, k={k}, std={noise_std}')
    plt.plot(ks, errors)
    plt.show()


from src.algorithms.utils import cryo_downsample


def exp4():
    data = np.random.normal(0, 1, (100, 100))
    # print(np.nanvar(data))
    data_downsampled = np.real(cryo_downsample(data, (10, 10)))
    # print(np.nanvar(data_downsampled))
    print(np.sqrt(np.prod(data.shape) / np.prod(data_downsampled.shape)))
    print(np.nanstd(data) / np.nanstd(data_downsampled))


if __name__ == '__main__':
    exp4()
    # basis = create_chebyshev_basis(101, 5)
    # # plt.imshow(-basis[1])
    # # plt.show()
    # # crea+-te_chebyshev_basis2(100, 5)
    # sim_data = DataSimulator2D(rows=1000,
    #                            columns=1000,
    #                            signal_length=100,
    #                            signal_power=1,
    #                            signal_fraction=1 / 7,
    #                            signal_gen=Shapes2D.sphere,
    #                            noise_std=0.1,
    #                            noise_mean=0,
    #                            apply_ctf=False)
    #
    # signal = sim_data.create_signal_instance()
    # # plt.imshow(signal)
    # # plt.show()
    #
    # xs = np.linspace(-1, 1, 100)
    # ys = np.sqrt(1 - np.square(xs))
    # plt.plot(ys)
    # plt.show()
    #
    # p = np.polynomial.chebyshev.Chebyshev.fit(xs, ys, 5)
    # print(p)
    #
    # est_signal = 0.64 * basis[0] - 0.417 * basis[1] - 0.071 * basis[2]
    #
    # for i in range(5):
    #     plt.plot(basis[i][:, 50])
    # plt.show()
    #
    # # plt.imshow(est_signal)
    # # plt.show()
    # diff = est_signal - signal
    # print(np.linalg.norm(diff))
    # plt.imshow(diff)
    # plt.show()
