from util.mfsc import *
from os.path import exists

mat = scipy.io.loadmat('data/TIDIGIT_train.mat')

# --------------------------
# Making spike trains for all train samples

# load files
handler = result_handler()
train_samples = handler.load_file('data/results_lib_train_digit.npy')

spike_trains = np.zeros([train_samples.shape[0], 2, 41*41])

for nr_sample, filtered_sample in enumerate(tqdm(train_samples, desc='Computing sample')):

    samplerate = 20000
    duration = len(filtered_sample)/samplerate
    number_of_rows = 41
    winstep = duration/number_of_rows

    # computing spikes for each time frame
    idx = np.argsort(filtered_sample)
    x = np.atleast_2d(filtered_sample)
    transformed = np.zeros(x.shape)
    for i, row in enumerate(x):
        transformed[i] = (((row - np.min(row)) * (30 - 0)) / (np.max(row) - np.min(row))) + 0

    # concatenating time frames into one spiketrain
    amt_timesteps = int(np.max(transformed[0])) #30
    amt_frames = transformed.shape[0] #41
    amt_fqbands = transformed.shape[1] #40

    c = 0 # for indexing spike_train
    t = 0 # for indexing x axis
    spike_train = np.zeros((2, amt_frames*amt_frames))

    for i, frequency_bands in enumerate(idx):
        spike_train[:, c:c+amt_fqbands] =\
            [transformed[i, :]+t, frequency_bands]
        c += amt_fqbands + 1
        t += amt_timesteps + 1

    # spike_train is the variable which stores a spike train for a sample
    # spike_trains stores all spike_train samples
    spike_trains[nr_sample] = spike_train

spike_trains = np.floor(spike_trains)

print(spike_trains.shape)

# save spike_trains to .npy
amt_samples = len(train_samples)
spike_trains01 = np.zeros((amt_fqbands, 31*41)) # n_timesteps * n_timeframes
all_spike_trains = np.zeros((amt_samples, spike_trains01.shape[0], spike_trains01.shape[1]))

# computing spike trains
if not exists('data/spike_trains_train.npy'):

    for i, spike_train in enumerate(tqdm(spike_trains, desc='Spike trains')):
        spike_trains01 = np.zeros((amt_fqbands, amt_frames*amt_frames))
        for f_band in range(amt_fqbands):
            arr = np.array([i for i, x in enumerate(spike_train[1]) if x == f_band])
            idx = spike_train[0][arr]
            idx = idx[idx != 0]
            st = spike_trains01[f_band,:] 
            idx = idx.astype(int)
            np.put(st, idx, 1)
            spike_trains01[f_band,:] = st
        all_spike_trains[i] = spike_trains01[:,0:1271]

    # Saving files
    handler = result_handler()
    handler.save_file('data/spike_trains_train.npy', all_spike_trains)
    print('Saved train data: ', all_spike_trains.shape)

