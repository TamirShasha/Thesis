import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line, ellipse, disk
from sklearn.mixture import GaussianMixture
import math
import scipy.stats as stats


def plot_circle_cut_length_hist(radius):
    n = 3 * radius
    img = np.zeros((n, n))
    rr, cc = disk((n // 2, n // 2), radius, shape=(n, n))
    img[rr, cc] = 1
    plot_cut_length_hist(img, 2 * radius)
    # plot_3_bins_cut_dist(img, 2 * radius)


def plot_ellipse_cut_length_hist(major_radius, minor_radius):
    n = 3 * major_radius
    img = np.zeros(shape=(n, n))
    r, c = n // 2, n // 2
    rr, cc = ellipse(r, c, major_radius, minor_radius)
    img[rr, cc] = 1
    plt.imshow(img)
    plt.show()
    # plot_cut_length_hist(img, 2 * major_radius)
    plot_3_bins_cut_dist(img, 2 * major_radius)


def get_cuts(img, total_cuts):
    n = img.shape[0]
    cuts = []
    m = total_cuts

    while m > 0:
        sides = np.random.choice(4, 2, replace=False)
        points = np.array([(0, np.random.randint(n)), (n - 1, np.random.randint(n)),
                           (np.random.randint(n), 0), (np.random.randint(n), n - 1)])
        first_point, second_point = points[sides]

        rr, cc = line(first_point[0], first_point[1], second_point[0], second_point[1])
        img[rr, cc] += 1

        intersection_len = np.sum((img == 2))
        if intersection_len > 0:
            cuts.append(intersection_len)
            m -= 1
            if m % 1000 == 0:
                print(m)

        img[rr, cc] -= 1

    cuts = np.array(cuts)
    cuts = cuts / np.max(cuts)
    return cuts


def plot_cut_length_hist(img, max_cut):
    n = img.shape[0]
    m = 1000
    num_of_bins = 50
    bin_size = max_cut // num_of_bins
    intersections = np.zeros(max_cut // bin_size)
    total_intersection_events = 0

    while m > 0:
        sides = np.random.choice(4, 2, replace=False)
        points = np.array([(0, np.random.randint(n)), (n - 1, np.random.randint(n)),
                           (np.random.randint(n), 0), (np.random.randint(n), n - 1)])
        first_point, second_point = points[sides]

        rr, cc = line(first_point[0], first_point[1], second_point[0], second_point[1])
        img[rr, cc] += 1

        intersection_len = np.sum((img == 2))
        if intersection_len > 0:
            intersections[intersection_len // bin_size] += 1
            total_intersection_events += 1
            m -= 1

        img[rr, cc] -= 1

    normed_intersections = np.array(intersections) / total_intersection_events
    cuts = [np.round(i * bin_size / max_cut, 2) for i in range(intersections.shape[0])]
    names = [f'{cut}' for cut in cuts]

    avg = (np.sum(
        (np.arange(1, intersections.shape[0] + 1) * bin_size) * intersections) / total_intersection_events) / max_cut

    print(names)
    print(normed_intersections)
    # plt.title(f'average: {avg}')
    # plt.bar(names, normed_intersections)
    # plt.show()

    # X = [[cuts[i], normed_intersections[i]] for i in range(len(cuts))]
    gmm = GaussianMixture(n_components=1, covariance_type='full')
    gm = gmm.fit(list(zip(cuts, normed_intersections)))
    print(gm.means_)
    print(np.sqrt(gm.covariances_))

    # mu = 0
    # variance = 1
    # sigma = math.sqrt(variance)
    # # x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    # # plt.plot(cuts, scipy.stats.norm.pdf(cuts, gm.means_[0], gm.covariances_[0]))

    plt.title(f'sumed: {avg}')
    # plt.bar(names, normed_intersections)
    plt.show()


def plot_3_bins_cut_dist(img, max_length, bars=None):
    if bars is None:
        bars = [0.4, 0.7]

    b1, b2 = bars
    n = img.shape[0]
    m = 1000
    intersections = np.zeros(3)

    counts = 0
    while counts < m:
        sides = np.random.choice(4, 2, replace=False)
        points = np.array([(0, np.random.randint(n)), (n - 1, np.random.randint(n)),
                           (np.random.randint(n), 0), (np.random.randint(n), n - 1)])
        first_point, second_point = points[sides]

        rr, cc = line(first_point[0], first_point[1], second_point[0], second_point[1])
        img[rr, cc] += 1

        intersection_len = np.sum((img == 2)) / max_length
        if intersection_len > 0:
            if intersection_len < b1:
                intersections[0] += 1
            elif intersection_len < b2:
                intersections[1] += 1
            else:
                intersections[2] += 1
            counts += 1

        img[rr, cc] -= 1

    normed_intersections = np.array(intersections) / m
    names = [f'0-{b1}', f'{b1}-{b2}', f'{b2}-1']

    print(normed_intersections)
    plt.bar(names, normed_intersections)
    plt.show()


def circle_img(radius):
    n = 3 * radius
    img = np.zeros((n, n))
    rr, cc = disk((n // 2, n // 2), radius, shape=(n, n))
    img[rr, cc] = 1
    return img


def gmm_cuts(img, total_cuts=10000):
    cuts = get_cuts(img, total_cuts).reshape(-1, 1)

    hx, hy, _ = plt.hist(cuts, bins=50, density=1, color="lightblue")
    plt.title('Gaussian mixture example 01')
    plt.grid()
    plt.show()

    gmm = GaussianMixture(n_components=3, covariance_type='full').fit(cuts)
    print(gmm.means_)
    print(gmm.covariances_)
    x = np.linspace(0, 1, 100)
    for mu, sig in zip(gmm.means_, np.sqrt(gmm.covariances_)):
        plt.plot(x, stats.norm.pdf(x, mu[0], sig[0][0]))
    plt.show()


def __main__():
    # plot_circle_cut_length_hist(300)
    # plot_ellipse_cut_length_hist(300, 100)
    circle = circle_img(300)
    gmm_cuts(circle)


if __name__ == '__main__':
    __main__()
