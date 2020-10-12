import os
import argparse
import glob
'''
The sample test data is in the wrong order.
Whereas the inputs 0-3 should be in the order:

context_word: [seq, 1,] of string
context_char: [seq, 1, 1, 16] of string
query_word: [seq, 1,] of string
query_char: [seq, 1, 1, 16] of string

Instead, the sample inputs are given in the order

cw
qw
cc
qc

So use this to swap.
'''

def swap_inside_dir(test_data_dir):
    '''
    Load tensor data from pb files in a single test data dir.
    :param test_data_dir: path to where the pb files for each input are found
    :return input data for the model
    '''
    input_1 = os.path.join(test_data_dir, 'input_1.pb')
    input_2 = os.path.join(test_data_dir, 'input_2.pb')

    os.rename(input_2, 'temp')
    os.rename(input_1, input_2)
    os.rename('temp', input_1)

def load_test_data(test_data_dir):
    tests = glob.glob(os.path.join(test_data_dir, 'test_data_set_*'))
    print("Got {} cases".format(len(tests)))
    [swap_inside_dir(d) for d in tests]
    print("Done")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, help="directory with test data")
    args = parser.parse_args()
    return args

args = get_args()
load_test_data(args.directory)