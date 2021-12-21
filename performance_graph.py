from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 8})
keys = ['FFNN', 'RNN-LSTM', 'ConvFF', 'ConvLSTM']

fig, axs = plt.subplots(2, 1)

axs[0].bar(keys, [14.8, 28.6, 6, 10.2], color=['r', 'b', 'r', 'b'])
axs[0].title.set_text("Memory usage - CPU")
axs[0].set_ylabel("Memory usage (MB)")
axs[0].set_xticks([])

axs[1].bar(keys, [3.8, 7.1, 0.9, 3.9], color=['r', 'b', 'r', 'b'])
axs[1].title.set_text("Memory usage - TPU")
axs[1].set_ylabel("Memory usage (MB)")
plt.xlabel("Models")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].bar(keys[:2], [435, 448], color=['r', 'b'])
axs[0].set_ylabel("Memory usage (MB)")

axs[1].bar(keys[2:4], [1.2, 3.3], color=['r', 'b'])
axs[1].set_ylabel("Memory usage (MB)")

plt.suptitle("Memory usage - GPU")
plt.tight_layout()
plt.show()
