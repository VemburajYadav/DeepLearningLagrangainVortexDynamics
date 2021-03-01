import matplotlib.pyplot as plt
from visualisations.my_plot import set_size
import numpy as np
import pandas as pd
import os

width = 455.24408

save_dir = '/home/vemburaj/Desktop/Ppt_Plots/Results/'

plt.style.use('tex')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

train_error = [72.52, 59.06, 46.63, 44.81, 33.59]
val_error = [76.97, 68.08, 52.17, 51.51, 36.41]
test_error = [82.34, 70.91, 55.18, 54.52, 37.98]


# index = ['VortexNet-0', 'VortexNet-1', 'VortexNet-2', 'VortexNet-3', 'Interaction-Network']
# df = pd.DataFrame({'Train error': train_error,
#                     'Val error': val_error,
#                    'Test error': test_error}, index=index)
# ax1 = df.plot.bar(rot=0, color={"Train error": "green", "Val error": "red", "Test error": "blue"})
# ax1.set_ylabel(r'Mean Squared Error')
#
data = [[72.52, 59.06, 46.63, 44.81, 33.59],
[76.97, 68.08, 52.17, 51.51, 36.41],
[82.34, 70.91, 55.18, 54.52, 37.98]]
X = np.arange(5)
fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
# ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
ax.legend(labels=['Train', 'Val', 'Test'])
ax.set_ylabel('Mean Squared Error')
ind = np.arange(5)
plt.xticks(ind + 0.25, ['VortexNet-0', 'VortexNet-1', 'VortexNet-2', 'VortexNet-3', 'Interaction-Network'])
fig.savefig(os.path.join('/home/vemburaj/Desktop', 'bar_plot_2.pdf'), format='pdf')
