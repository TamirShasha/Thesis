import numpy as np


class Experiment:
    def __init__(self, file_name):
        self._load_experiment(file_name)

    def _load_experiment(self, file_name):
        print(f'Loading experiment from {file_name}')
        self._data_length = 10
        pass

    def plot_likelihoods(self):
        pass

    def plot_rms(self):
        pass
