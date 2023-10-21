import matplotlib.pyplot as  plt

from src.utils.common_filter_basis import create_filter_basis

save_path = r'C:\Users\tamir\Desktop\Experiments For Paper\chebyshev_2d'

filter_basis = create_filter_basis(100, 8)
for i in range(2):
    for j in range(4):
        elem = filter_basis[4 * i + j]
        plt.imshow(elem, cmap='Blues')
        plt.tight_layout()
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        plt.savefig(f'{save_path}\elem_{4 * i + j}.png')
        plt.show()
plt.show()
