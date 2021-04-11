import numpy as np


class DynamicProgramming2D:
    width = 1000
    height = 1000
    occourences = 2
    d_width = 1
    d_height = 1

    def __init__(self):
        pass

    def dp_single_row(self, occurences):
        if occurences == 0:
            return 1

        if occurences * self.d_width > self.width:
            return 0

        mapping = np.zeros(shape=(self.width + 1, occurences + 1))
        mapping[:, 0] = 1
        for k in np.arange(occurences) + 1:
            for i in np.flip(np.arange(self.width - self.d_width + 1)):
                num_of_available_cells = self.width - i
                num_of_required_cells = k * self.d_width

                if num_of_required_cells > num_of_available_cells:
                    mapping[i, k] = 0
                else:
                    mapping[i, k] = mapping[i + 1, k] + mapping[i + self.d_width, k - 1]

        return mapping[0, occurences]

    def dp_multilpe_rows(self):
        mapping = np.zeros(shape=(self.height + 1, self.occourences + 1))
        mapping[-1, 0] = 1
        for row in np.flip(np.arange(self.height)):  # start from last row
            for k in np.arange(self.occourences + 1):  # for each num of left occourences

                for _k in np.arange(k + 1):
                    # _k go to current row, the rest goes to the rest
                    term1 = self.dp_single_row(_k)
                    term2 = mapping[row + 1, k - _k]
                    mapping[row, k] += term1 * term2
        print(mapping[0, self.occourences])


dp = DynamicProgramming2D()

dp.dp_multilpe_rows()
