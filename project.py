from util.mfsc import *

# load files
handler = result_handler()
train = handler.load_file('data/results_lib_train_digit.npy')
test = handler.load_file('data/results_lib_test_digit.npy')

# unfiltered training data (for visualisation)
mat = scipy.io.loadmat('data/TIDIGIT_train.mat')

interesting = mat['train_samples'][1,0]
audio = [item for sublist in interesting for item in sublist]

samplerate = 20000
duration = len(interesting)/samplerate
number_of_rows = 41
winstep = duration/number_of_rows
new_result = logfbank(interesting , samplerate = samplerate , winlen = 2 * winstep, winstep = winstep, nfilt=40, nfft=1024)

# plot
plt.figure(figsize=(15,4))
plt.subplot(1, 3, 1)
plt.plot(np.linspace(0, len(interesting) / samplerate, num=len(interesting)), audio)
plt.imshow(new_result, aspect='auto', origin='lower')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Frequency spectrum")

samplerate = 20000
duration = len(train[0])/samplerate
number_of_rows = 41
winstep = duration/number_of_rows
new_result = logfbank(train[0] , samplerate = samplerate , winlen = 2 * winstep, winstep = winstep, nfilt=40, nfft=1024)

plt.subplot(1, 3, 2)
plt.plot(np.linspace(0, len(train[0]) / samplerate, num=len(train[0])), train[0])
plt.imshow(new_result, aspect='auto', origin='lower')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel("Time")
plt.ylabel("MFSC features")
plt.title("MFSC")

plt.subplot(1, 3, 3)
plt.hist(train[0])
plt.xlabel("MFSC feature")
plt.ylabel("Count")
plt.title("Count of MFSC features")
plt.tight_layout()
plt.show()
