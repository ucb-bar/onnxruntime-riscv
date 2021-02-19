import os
import sys
import numpy as np
import glob
import re
import abc
import subprocess
import json
import argparse
import time
from PIL import Image

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from quantizer.quantize import quantize_static, CalibrationDataReader, QuantFormat, QuantType

from data_preprocess import load_batch, preprocess_mxnet_raw, \
                            preprocess_caffe_raw, preprocess_caffe2_raw, \
                            preprocess_rcnn_raw

class TrainingDataReader(CalibrationDataReader):
    def __init__(self, pb_folder, max_to_load, preprocess_method, augmented_model_path='augmented_model.onnx'):
        self.image_folder = pb_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_inputs = True
        self.max_to_load = max_to_load
        self.preprocess_method = preprocess_method
        self.data_dicts = []

    def get_next(self):
        if self.preprocess_inputs:
            self.preprocess_inputs = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            num_inputs_for_model = len(session.get_inputs())

            test_data = load_test_data(self.image_folder, num_inputs_for_model, self.max_to_load, self.preprocess_method)
            print('Num cases {}, num inputs for each cases {}'.format(
                len(test_data), num_inputs_for_model))
            input_names = [i.name for i in session.get_inputs()]
            data_dicts = [{input_names[i]: datum[i] for i in range(len(input_names))} for datum in test_data]
            self.enum_data_dicts = iter(data_dicts)
        return next(self.enum_data_dicts, None)


def load_single_test_data(test_data_dir, num_expected_inputs,
                          preprocess_method):
    '''
    Load tensor data from pb files in a single test data dir.
    :param test_data_dir: path to where the pb files for each input are found
    :return input data for the model
    '''
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    if inputs_num != num_expected_inputs:
        raise ValueError(
            'Number of input protobufs does not match expected model inputs')
    if not inputs_num:
        raise ValueError('No protobufs found in test data directory')
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        tensor = numpy_helper.to_array(tensor)
        if preprocess_method == 'mxnet':
            tensor = preprocess_mxnet_raw(tensor)
        elif preprocess_method == 'caffe':
            tensor = preprocess_caffe_raw(tensor)
        elif preprocess_method == 'caffe2':
            tensor = preprocess_caffe2_raw(tensor)
        elif preprocess_method == 'rcnn':
            tensor = preprocess_rcnn_raw(tensor)
        inputs.append(tensor)
    return inputs

def load_test_data(test_data_dir, num_expected_inputs, max_to_load,
                   preprocess_method):
    tests = glob.glob(os.path.join(test_data_dir, 'test_data_set_*'))
    if len(tests):
        return [load_single_test_data(case, num_expected_inputs, preprocess_method) for case in \
                (tests[:max_to_load] if max_to_load else tests)]
    else:
        return [load_single_test_data(test_data_dir, num_expected_inputs, preprocess_method)]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_model_path',
                        type=str,
                        default='calibrated_quantized_model.onnx')
    parser.add_argument(
        '--dataset_size',
        type=int,
        default=0,
        help=
        "Number of images or tensors to load. Default is 0 which means all samples"
    )
    parser.add_argument(
        '--data_preprocess',
        type=str,
        required=True,
        choices=[
            'mxnet', 'caffe', 'caffe2', 'rcnn', 'None'
        ],
        help="Refer to Readme.md for guidance on choosing this option.")
    parser.add_argument('--mode',
                        type=str,
                        required=False,
                        choices=['int8', 'uint8'],
                        default='int8',
                        help="Whether to quantize in int8 or uint8.")
    parser.add_argument('--static',
                        required=True,
                        type=lambda x:
                        (str(x).lower() in ['true', '1', 'yes']))
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.model_path
    output_model_path = args.output_model_path
    if args.static:
        calibration_dataset_path = args.dataset_path
        dr = TrainingDataReader(calibration_dataset_path, args.dataset_size, args.data_preprocess)
        quantize_static(input_model_path,
                        output_model_path,
                        dr,
                        optimize_model=False,
                        quant_format=QuantFormat.QOperator,
                        per_channel=False,
                        activation_type=QuantType.QInt8 if args.mode == 'int8' else QuantType.QUInt8,
                        weight_type=QuantType.QInt8 if args.mode == 'int8' else QuantType.QUInt8)
        print('Calibrated and quantized model saved.')
    else:
        quantize_dynamic(input_model_path,
                        output_model_path,
                        optimize_model=False,
                        per_channel=False,
                        activation_type=QuantType.QInt8 if args.mode == 'int8' else QuantType.QUInt8,
                        weight_type=QuantType.QInt8 if args.mode == 'int8' else QuantType.QUInt8)
        print('Dynamically quantized model saved.') 


if __name__ == '__main__':
    main()