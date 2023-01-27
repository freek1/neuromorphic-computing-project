# Neuromorphic Computing Project

The aim of the project is to replicate the spoken digit classification SNN described by Dong et al. in Nengo.
This project is inspired by earlier done research by Dong et al., which showed high performance in two different situations (97.5\% and 93.8\% resp.).

Dong M, Huang X, Xu B (2018) Unsupervised speech recognition through spike-timing-dependent plasticity in a convolutional spiking neural network. PLoS ONE 13(11): e0204596. https://doi.org/10.1371/journal.pone.0204596

## Preprocessing of the data
To convert the `.mat` TIDIGIT database to the `.npy` format this project uses, run the following files:

1. Make sure you have downloaded `TIDIGIT_train.mat` and `TIDIGIT_test.mat` in the data folder.
2. Run `util/mat2npy.py` to convert the `.mat` files into preprocessed (and MFSC filtered) `.npy` data (train and test, data and labels) files.
    - Creates: `data/results_lib_train_digit.npy`, `data/results_lib_test_digit.npy`, `data/train_labels.npy` and `data/test_labels.npy`
3. Run `util/spike_trains_train.py` to convert the preprocessed `.npy` file into spike trains
    - Creates: `data/spike_trains_train.npy`
4. Run `util/spike_trains_test.py` to convert the preprocessed `.npy` file into spike trains
    - Creates: `data/spike_trains_test.npy`
5. Optional: To create the figures to visualize the preprocessing stages, run `visualize_preprocess.py`

## Running the model
To run the Nengo model and do the classification on the data, simply run `nengo_classification.py` (this may take a few minutes). The LIMIT variable may be adjusted to change the amount of samples used in the model, depending on computational strength of your device.

## Processing audio into spikes
The audio files are preprocessed by passing them through a MFSC filter making for 41 frames with 40 frequency bands per sample, resulting in e.g. figure 1A below. The spikes for each frame are computed by time-to-first-spike encoding for each frequency band in 30 time steps, i.e. a higher activity in a frequency band in a frame results in a spike at an earlier time step for that frequency band, see figure 1B.

![Figure 1](/figures/mfsc_spectogram_spike_coding_one-1.png "Figure 1")

Doing this for every frame, results in spike encodings as seen for three frames in figure 2.

![Figure 2](/figures/spike_coding_one_5-16-25-1.png "Figure 2")

Finally, concatenating all 41 frames in order for each sample, results in spike encodings of the spoken digits for all samples. Figure 3 shows three examples.

![Figure 3](/figures/spiketrains_10-14-1105-1.png "Figure 3")

## The Nengo model & classification
The model has the following architecture, as can be seen in Figure 4.

![Figure 4](/figures/model_architecture.png "Figure 4")

Prior to the Nengo model, the data is pre-processed with a convolution step. Then, all samples are fed into the pre node, after which the learning takes place. The weights are then taken from the learning connection to pool together, and finally this pooled data is used for the classification.
