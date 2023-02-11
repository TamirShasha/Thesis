import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

main_dir = r'C:\Users\tamir\Desktop\Experiments For Paper\Repeated Experiments\Data_5'
files = os.listdir(main_dir)

data = []
for file in files[:5]:
    data_path = os.path.join(main_dir, file, 'data.pkl')
    data.append(pickle.load(open(data_path, 'rb')))

all_likelihoods = np.array([_data['likelihoods'] for _data in data])
sizes_options = data[0]['sizes_options']

mean = np.mean(all_likelihoods, axis=0)
std = np.std(all_likelihoods, axis=0)

plt.title(f"Total of {len(data)} experiments, Noise\u007E\u2115({data[0]['noise_mean']}, {data[0]['noise_std']})")
plt.plot(sizes_options, mean, 'k-')
plt.fill_between(sizes_options, mean - std, mean + std, alpha=0.5)
plt.show()
