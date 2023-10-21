import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

main_dir = r'C:\Users\tamir\Desktop\Experiments For Paper\Number _Of_Instances\Data'
files = os.listdir(main_dir)

data = []
for file in files[:5]:
    data_path = os.path.join(main_dir, file, 'data.pkl')
    data.append(pickle.load(open(data_path, 'rb')))

plt.set_cmap("Greens")
for _data in data:
    ys = _data['likelihoods']
    ys = ys / np.linalg.norm(ys)
    plt.plot(_data['sizes_options'], ys, 'o-', label=f"k={_data['number_of_instances'][0]}")
plt.axvline(x=data[0]['signal_size'], linestyle='--', color='black')
plt.legend()
plt.show()
print()
