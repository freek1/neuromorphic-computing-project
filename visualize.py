from util.mfsc import *
from os.path import exists

# load files
handler = result_handler()
spike_trains_train = handler.load_file('data/spike_trains_train.npy')

# 30 * 41 = 1230
n_samples = 2464
n_freqbands = 40
n_timeframes = 41
n_timesteps = 30

nengo_shape = np.reshape(spike_trains_train, (2464, 40, 41, 30))

print(nengo_shape.shape)

fig, ax = plt.subplots(1, 12, sharex=True, sharey=True)
for i in range(12):
    ax[i].imshow(nengo_shape[0, :, i, :], cmap='gray_r')
    ax[i].set_title(i)
plt.show()

# You should acces the nengo_shape matrix as follows:
# nengo_shape[i_sample, :, i_frame, :]
# Where i_sample is the number of the (training) sample
# And i_frame is the number of the frame (out of 41)
