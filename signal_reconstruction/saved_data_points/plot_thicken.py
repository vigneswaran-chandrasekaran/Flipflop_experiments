from matplotlib import pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 22})

data = np.load('data.npy')
#f = np.load('ff_pred_010.npy')
d = data[1]
plt.plot(d, linewidth=4, label='original')
#plt.plot(f[0], linewidth=4, label='generated')
plt.title('Class - 001')
#plt.legend(loc='upper right')
plt.show()
