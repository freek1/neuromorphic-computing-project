from util.mfsc import *
from os.path import exists

# load files
handler = result_handler()
spike_trains_train = handler.load_file('data/spike_trains_train.npy')

# 31 * 41 = 1271
n_samples = 2464
n_freqbands = 40
n_timeframes = 41
n_timesteps = 31 # actually 30 but then it doesnt fit??

nengo_shape = np.reshape(spike_trains_train, (2464, 40, 41, 31))

print(nengo_shape.shape)

plt.figure(figsize=(5,5))
plt.imshow(nengo_shape[0, :, 0, :], cmap='gray_r')
plt.show()

# You should acces the nengo_shape matrix as follows:
# nengo_shape[i_sample, :, i_frame, :]
# Where i_sample is the number of the (training) sample
# And i_frame is the number of the frame (out of 41)
