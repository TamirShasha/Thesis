import seaborn as sns
import matplotlib.pyplot as plt

results = [
    [0, 1, 2],
    [1, 2, 3]
]

sns.heatmap(results, annot=True)
plt.show()