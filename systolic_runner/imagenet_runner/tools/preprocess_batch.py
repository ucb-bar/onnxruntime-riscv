'''
Tool to batch preprocess the ILSVRC2012 set and generate the 224x224 crops needed for inference
Outputs a file that can be passed to the imagenet runner

file_to_class.txt contains the classes for the validation set
    with line 1 corresponding to ILSVRC2012_val_00000001, etc.

class_to_label.txt contains the output label each class maps to,
    with first line corresponding to output label 0

`input_folder` should point directly to the validation set
At the end, `batch_out` will be written to disk, which can be passed to runner

'''

import argparse
import numpy as numpy
import glob
import pathlib
import random
import os
import cv2
from multiprocessing import Pool, Process, Array
from pathlib import Path
import tqdm


def filename_to_num(filename):
    '''
    Convert filename to the filenumber.
    E.g. ILSVRC2012_val_00000001 -> 1
    '''
    return int(filename.split("_")[-1].split(".")[0])


def preprocess(filename):
    filename_idx = filename_to_num(filename)
    imagenet_class = file_to_class[filename_idx]
    label = class_to_label[imagenet_class]

    img_file_path = os.path.join(args.output_folder, label)
    Path(img_file_path).mkdir(parents=True, exist_ok=True)
    img_file_path = os.path.join(img_file_path, str(filename_idx) + '.jpg')

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    # Resize
    height, width, _ = img.shape
    new_height = height * 256 // min(img.shape[:2])
    new_width = width * 256 // min(img.shape[:2])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Crop
    height, width, _ = img.shape
    startx = width//2 - (224//2)
    starty = height//2 - (224//2)
    img = img[starty:starty+224,startx:startx+224]
    assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)

    cv2.imwrite(img_file_path, img)

    return img_file_path


parser = argparse.ArgumentParser(description='ImageNet batch preprocess')
parser.add_argument('--sample', type=int)
parser.add_argument('--input_folder', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--processes', default = 8, type=int)
args, args_other = parser.parse_known_args()

input_files = glob.glob(os.path.join(args.input_folder, '*.JPEG'))

if (args.sample):
    input_files = random.sample(input_files, args.sample)

print("Pre-processing {} files".format(len(input_files)))

class_to_label = []
with open("class_to_label.txt") as f:
    class_to_label = [line.rstrip('\n') for line in f]
class_to_label = {value: str(idx) for (idx, value) in enumerate(class_to_label)}

file_to_class = []
with open("file_to_class.txt") as f:
    file_to_class = [line.rstrip('\n') for line in f]
file_to_class = {idx + 1: value for (idx, value) in enumerate(file_to_class)}


output_paths = []

with Pool(processes=8) as pool:
    for ret_path in tqdm.tqdm(pool.imap_unordered(preprocess, input_files), total=len(input_files)):
        output_paths.append(ret_path)

with open("batch_out.txt", "w") as outfile:
    outfile.write("\n".join(output_paths) + "\n")
