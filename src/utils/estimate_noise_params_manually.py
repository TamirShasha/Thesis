import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.utils.micrograph import Micrograph


# Load Image
def load_micrograph(file_path: str):
    return Micrograph(file_path=file_path,
                      clip_outliers=True)


# Draw Chosen Rectangles
def calc_statistics(file_path, idxs):
    mrc = load_micrograph(file_path=file_path)
    data = mrc.get_micrograph()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(data, cmap='gray')

    mrc_patches = []
    for idx in idxs:
        rect = patches.Rectangle(idx[:2], idx[2], idx[2], color='red', fill=False)
        axs[0].add_patch(rect)
        patch = data[idx[0]: idx[0] + idx[2], idx[1]: idx[1] + idx[2]]
        mrc_patches.append(np.array(patch).flatten())

    patches_data = np.concatenate(mrc_patches)

    data_std = np.round(np.std(data), 3)
    data_mean = np.round(np.mean(data), 3)
    noise_std = np.round(np.std(patches_data), 3)
    noise_mean = np.round(np.mean(patches_data), 3)

    axs[1].hist(patches_data, bins=50, alpha=0.5, density=True)
    axs[1].hist(data.ravel(), bins=50, alpha=0.5, density=True)
    plt.show()

    print(data_mean, data_std)
    print(noise_mean, noise_std)
# Estimate Noise
# Draw Histograms.


if __name__ == '__main__':
    # file_path = r'C:\Users\tamir\Desktop\Thesis\data\EMPIAR_10128\001.mrc'
    # chosen_idxs = np.array([
    #     [2286, 640, 400],
    #     [350, 840, 200],
    #     [1200, 2000, 200],
    #     [1864, 2141, 200],
    #     [1113, 3026, 150]
    # ])
    # calc_statistics(file_path=file_path, idxs=chosen_idxs)

    file_path = r'C:\Users\tamir\Desktop\Thesis\data\10028\005.mrc'
    chosen_idxs = np.array([
        [1290, 773, 300],
        [2500, 1200, 400],
        [1284, 2770, 300],
        [215, 3283, 300],
        # [1113, 3026, 150]
    ])
    calc_statistics(file_path=file_path, idxs=chosen_idxs)

