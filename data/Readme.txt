This database converts the waveforms from speech corpus TIDIGITS into spike patterns.

1. The dataset is devided into training set and testing set.
2. The spike patterns are saved in the form of exact firing timings, in the unit of ms.
3. For more encoding details, please refer to our paper: https://arxiv.org/pdf/1909.01302.pdf
4. For more information of the original TIDIGITS database, please refer to the manual of TIDIGITS from: https://catalog.ldc.upenn.edu/LDC93S10

INSTRUCTIONS:
To convert the .mat TIDIGIT database to the .npy format this project uses, run the following files:

1. Make sure you have downloaded TIDIGIT_train.mat and TIDIGIT_test.mat in the data folder.
2. Run util/mat2npy.py to convert the .mat files into preprocessed (and MFSC filtered) .npy data (train and test, data and labels) files.
    - Creates: data/results_lib_train_digit.npy, data/results_lib_test_digit.npy, data/train_labels.npy and data/test_labels.npy
3. Run util/spike_trains_train.py to convert the preprocessed .npy file into spike trains
    - Creates: data/spike_trains_train.npy
4. Run util/spike_trains_test.py to convert the preprocessed .npy file into spike trains
    - Creates: data/spike_trains_test.npy
5. Optional: To create the figures to visualize the preprocessing stages, run visualize_preprocess.py
