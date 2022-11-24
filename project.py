from util.mfsc import *

# load files
handler = result_handler()
train = handler.load_file('data/results_lib_train_digit.npy')
test = handler.load_file('data/results_lib_test_digit.npy')

print('train', train.shape)
print('test', test.shape)

# unfiltered training data (for visualisation)
mat = scipy.io.loadmat('data/TIDIGIT_train.mat')

train_sample_0 = mat['train_samples'][1,0]
audio = [item for sublist in train_sample_0 for item in sublist]

samplerate = 20000
duration = len(train_sample_0)/samplerate
number_of_rows = 41
winstep = duration/number_of_rows
new_result = logfbank(train_sample_0 , samplerate = samplerate , winlen = 2 * winstep, winstep = winstep, nfilt=40, nfft=1024)

# plot fig 2 from Dong et al
plt.figure(figsize=(9,4))
plt.tight_layout()
plt.suptitle('Input coding of SNN for spoken digit "One"')

# freq bands vs frames
plt.subplot(1, 2, 1)
plt.imshow(new_result, aspect='auto', origin='lower')
# plt.colorbar()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel("Frequency bands")
plt.ylabel("Frames")
plt.title("$\mathbf{A}$")

# freq bands vs time steps (at frame = 17)
plt.subplot(1, 2, 2)

idx = np.flip(np.argsort(new_result[17, :]))
x = np.atleast_2d(new_result[17, :])
transformed = x + np.min(x) # shift all values to positive [0, ->]
downscaled = transformed/np.max(transformed) - 1 # scaled to range [0, 1]
upscaled = downscaled * 30 * 2 # scaled to range [0, 30]

plt.scatter(np.floor(upscaled), idx)
plt.xlim(0, 30)
plt.xlabel("Time steps")
plt.ylabel("Frequency bands")
plt.title("$\mathbf{B}$")

# save figure
plt.savefig('figures/mfsc_spectogram_spike_coding_one.pdf', dpi=1000, format='pdf')

plt.show()


