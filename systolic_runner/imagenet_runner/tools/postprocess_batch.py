import argparse
import numpy as numpy
import glob
import os
import pathlib


def process_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = lines[-3:]

    assert lines[0].strip() == 'Finished batch', "Did not find finished message in split {}".format(file)
    top_five_right, total = lines[1].strip().split(':')[1].strip().split('/')
    top_one_right, total2 = lines[2].strip().split(':')[1].strip().split('/')
    assert total == total2, "Total mismatch"
    return int(top_five_right), int(top_one_right), int(total)

parser = argparse.ArgumentParser(description='ImageNet batch postprocess')
parser.add_argument('--input_folder', type=str, required=True)
args, args_other = parser.parse_known_args()

input_files = glob.glob(os.path.join(args.input_folder, 'out.*'))

total_total = 0
total_top_five_right = 0
total_top_one_right = 0

for f in input_files:
    top_five_right, top_one_right, total = process_file(f)
    total_total += total
    total_top_five_right += top_five_right
    total_top_one_right += top_one_right

print("Total lines processed {}\n".format(total_total))
print("Top five right: {}/{}".format(total_top_five_right, total_total))
print("Top one right: {}/{}".format(total_top_one_right, total_total))


