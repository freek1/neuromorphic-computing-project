from util.mfsc import *
from os.path import exists

mat = scipy.io.loadmat('data/TIDIGIT_train.mat')

# --------------------------
# Making spike trains for all train samples

# load files
handler = result_handler()
train_samples = handler.load_file('data/results_lib_train_digit.npy')

spike_trains = np.zeros([train_samples.shape[0], 2, 41*41])

for nr_sample, filtered_sample in enumerate(tqdm(train_samples)):

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

# plotting three examples of spiketrains
exs = [10, 14, 1105]
plt.figure(figsize=(14, 8)) 
plt.suptitle('Spoken digits converted to spike trains')
for p, ex in enumerate(exs):
    plt.subplot(3, 1, p+1)
    plt.scatter(spike_trains[ex][0], spike_trains[ex][1])
    plt.xlabel('Time steps')
    plt.ylabel('Frequency bands')
    plt.title(f"Train sample {ex}")

plt.tight_layout()

# save figure
if not exists(f'figures/spiketrains_10-14-1105.pdf'):
    plt.savefig('figures/spiketrains_10-14-1105.pdf', dpi=1000, format='pdf')

plt.show()
