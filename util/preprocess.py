from mfsc import *

'''
Running this file loads train and test data from data/ folder and processes it via MFSC.
Then the processed files are saved in data/ with .npy extentions.
'''

digit_converter = TIDIGIT_Converter()

# First test with 5 samples
results_own = digit_converter.convert_tidigit_own('data/TIDIGIT_train.mat', 'train_samples', 20000, 41, 40)
results_own = np.array(results_own)

# print(results_own.shape)

# Now, converting the train and test sets:
results_lib_train_digit = digit_converter.convert_tidigit_lib('data/TIDIGIT_train.mat', 'train_samples', 20000, 41, 40)
results_lib_test_digit = digit_converter.convert_tidigit_lib('data/TIDIGIT_test.mat', 'test_samples', 20000, 41, 40)

results_lib_train_digit = np.array(results_lib_train_digit)
results_lib_test_digit = np.array(results_lib_test_digit)

print('new train shape:', results_lib_train_digit.shape)
print('new test shape:', results_lib_test_digit.shape)

# Saving files
handler = result_handler()
handler.save_file('data/results_lib_train_digit.npy', results_lib_train_digit)
print('saved train')
handler.save_file('data/results_lib_test_digit.npy', results_lib_test_digit)
print('saved test')

