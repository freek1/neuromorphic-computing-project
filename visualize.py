from util.mfsc import *
from os.path import exists
import pickle

# load files
handler = result_handler()
spike_trains_train = handler.load_file('data/spike_trains_train.npy')

# pickle_out = open("data/spike_trains_train.pickle", "wb")
# pickle.dump(spike_trains_train, pickle_out)
# pickle_out.close()

plt.figure(figsize=(10, 15))
plt.imshow(spike_trains_train[0])
plt.show()
