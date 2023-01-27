# Imports
import nengo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split

from STDP_learning import STDP
from convolution import conv3D

# Parameters from the paper
n_freqbands = 40
n_timeframes = 41
n_timesteps = 30

n_neurons = n_freqbands * n_timeframes * n_timesteps

n_samples = 2464

f_maps = 50

window_size = [6,n_freqbands]
conv_inp_shape = (n_freqbands, n_timeframes, n_timesteps)
stride = [1,1]

mean = 0.8
std = 0.05

LIMIT = 650

presentation_time = 0.001

threshold = 23
thresh_config = nengo.presets.ThresholdingEnsembles(threshold) # Set the threshold

# Loading data
train = np.load("spike_trains_train_fixed2.npy")

train_conv = np.reshape(train, (40,41,30,2464))
train_new = []

for n in range(n_samples):
    train_new.append(conv3D(train_conv[:,:,:,n], np.array([[40,6]])))
    
train_new = np.array(train_new)
train_new = np.reshape(train_new.T, (1600,2464))[:,0:LIMIT]

%%time

model = nengo.Network(label="Audio STDP learning")

with model:

    # Layers ----------------------------------------------------------------------
    input_layer = nengo.Node(nengo.processes.PresentInput(train_new, presentation_time))
    
    pre = nengo.Ensemble(LIMIT, dimensions=LIMIT)
    
    post = nengo.Ensemble(LIMIT, dimensions=LIMIT)
    
    # Connections -----------------------------------------------------------------
    
    learn_conn = nengo.Connection(
        pre, post,
#         learning_rule_type = nengo.BCM(learning_rate=5e-10), # Change this later
        learning_rule_type = STDP(learning_rate=0.04),
        solver = nengo.solvers.LstsqL2(weights=True)
    )
    
    # Probes ----------------------------------------------------------------------
    
    synapses_probe = nengo.Probe(learn_conn,"weights",label="synapses")
    
print("Running for {} seconds".format(len(train)*presentation_time))
with nengo.Simulator(model) as sim:
        
    sim.run(len(train)*presentation_time) 

sample_output = sim.data[synapses_probe]
avg = sum(sample_output)/sample_output.shape[1]
pooled = block_reduce(avg, (4,1), np.max)

X = pooled.T

y = np.load("train_labels.npy",allow_pickle=True)
y = y[0:LIMIT]
y_new = []
for i in range(len(y)):
    y_new.append(y[i][0][0][0])
y_new = np.array(y_new)

X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size=0.2)

X_train=X_train.astype('int')
X_test=X_test.astype('int')
y_train=y_train.astype('int')
y_test=y_test.astype('int')

clf = LinearSVC()
clf.fit(X_train, y_train)

print("Accuracy with sample size" + str(LIMIT) ": " + str(clf.score(X_test, y_test)))