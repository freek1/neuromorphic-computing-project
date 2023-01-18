
# Imports
import nengo
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

# Load the data and reshape it into a form Nengo accepts
train = np.load("spike_trains_train.npy")
train = np.reshape(train, (30, 40*41, 2464)) # Reshape into [n_timesteps, n_freqbands * n_timeframes, n_samples] 

# Take only the first sample as a demo
input = train[:,:,0]

# Parameters from the paper ----------------------------------------------------------------------
n_freqbands = 40
n_timeframes = 41
n_neurons = n_freqbands * n_timeframes
n_timesteps = 30
n_samples = 2464
f_maps = 50
window_size = [6]

mean = 0.8
std = 0.05

threshold = 23
thresh_config = nengo.presets.ThresholdingEnsembles(threshold) # Set the threshold in nengo


# The model --------------------------------------------------------------------------------------
model = nengo.Network()

with model:
    
    # print(input[0].size)
    
    input_layer = nengo.Node(nengo.processes.PresentInput(input, 0.1), size_out=40*41) # n_neurons
    
    pre = nengo.Ensemble(1640, dimensions = 35*f_maps)
    
    post = nengo.Ensemble(1640, dimensions = 35*f_maps)
    
    transform = nengo.Convolution(
                n_filters = f_maps,
                input_shape = (40,41),
                kernel_size = window_size,
                strides = [1],
                padding="valid", # previously: "same"
                channels_last = True,
                init = nengo.dists.Gaussian(mean, std)
            )
    
    # And then you can apply it on the input as preprocessing step
    conv_conn = nengo.Connection(input_layer, pre, transform = transform)
    
    learn_conn = nengo.Connection(
        pre, post,
        learning_rule_type = STDP(learning_rate=6e-12),
        solver = nengo.solvers.LstsqL2(weights=True)
    )
    
    #Probes
    input_probe = nengo.Probe(input_layer)
    pre_probe = nengo.Probe(pre, synapse=0.01)
    post_probe = nengo.Probe(post, synapse=0.01)
    
    with nengo.Simulator(model) as sim:
        sim.run(time_in_seconds=5)

# Pooling ----------------------------------------------------------------------------------------

sample_output = sim.data[post_probe]
avg = sum(sample_output)/sample_output.shape[1]

pooled = block_reduce(avg, (4), np.max)