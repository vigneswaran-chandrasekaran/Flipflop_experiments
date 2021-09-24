from matplotlib import pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 22})

data = np.load('data.npy')
f = np.load('lstm_pred_010.npy')
d = data[17]
plt.plot(d, linewidth=4, label='original', color='r')
plt.plot(f[0], linewidth=4, label='generated')
plt.title('Class - 010')
plt.legend(loc='upper right')
plt.show()
