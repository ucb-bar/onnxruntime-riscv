#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import sys
import glob
import argparse
import copy
import numpy as np
from PIL import Image
import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from quantizer.quantize import quantize, QuantizationMode
from data_preprocess import load_batch, preprocess_mxnet_raw, \
                            preprocess_caffe_raw, preprocess_caffe2_raw, \
                            preprocess_rcnn_raw

import re
import subprocess
import json
import math

# Candidate nodes for quantization. Calibration will be done for these nodes only
# When more nodes are extended to support quantization, add them to this list
# Values are the relevant input indices that should be quantized
QUANTIZATION_CANDIDATES = {'Conv': [0], 'ConvTranspose': [0], 'MatMul': [0, 1], 'Attention': [0, 1],
                           'MaxPool': [0], 'AveragePool': [0], 'ScatterElements': [0]}
# Binary ops that need to be checked for floating-point before quantizing
BINARY_OPS_TO_QUANTIZE = ['Add', 'Mul']


def get_input_name_to_nodes(model):
    '''
        Helper function to create input_name_to_nodes dictionary
    '''
    input_name_to_nodes = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in input_name_to_nodes:
                input_name_to_nodes[input_name] = [node]
            else:
                input_name_to_nodes[input_name].append(node)
    return input_name_to_nodes

def can_quantize_name(node, name, value_infos, idx = None):
    if node.op_type in QUANTIZATION_CANDIDATES and (not idx or idx in QUANTIZATION_CANDIDATES[node.op_type]):
        return True
    if node.op_type in BINARY_OPS_TO_QUANTIZE and name in value_infos.keys():
        vi = value_infos[name]
        if vi.type.HasField('tensor_type') and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            return True
    return False


def create_reduce_nodes(edge_to_reduce, added_nodes, added_outputs,
                        reduced_edges, value_infos):
    # When doing ReduceMax/ReduceMin, keep dimension if tensor contains dim with value of 0,
    # for example:
    #     dim = [ dim_value: 0 ] 
    #  
    # otherwise, don't keep dimension. 
    #
    keepdims = 0
    shape = ()
    if edge_to_reduce in value_infos:
        dim = value_infos[edge_to_reduce].type.tensor_type.shape.dim
        for d in dim:
            # A dimension can be either an integer value or a symbolic variable.
            # Dimension with integer value and value of 0 is what we are looking for to keep dimension. 
            # Please see the def of TensorShapeProto https://github.com/onnx/onnx/blob/master/onnx/onnx.proto#L630
            if d.WhichOneof('value') == 'dim_value' and d.dim_value == 0:
                keepdims = 1
                shape = (1,) if len(dim) == 1 else list(1 for i in range(len(dim)))
                break

    # Adding ReduceMin nodes
    reduce_min_node = onnx.helper.make_node('ReduceMin', [edge_to_reduce],
                                            [edge_to_reduce + '_ReduceMin'],
                                            keepdims=keepdims)
    added_nodes.append(reduce_min_node)
    added_outputs.append(
        helper.make_tensor_value_info(reduce_min_node.output[0],
                                      TensorProto.FLOAT, shape))

    # Adding ReduceMax nodes
    reduce_max_node = onnx.helper.make_node('ReduceMax', [edge_to_reduce],
                                            [edge_to_reduce + '_ReduceMax'],
                                            keepdims=keepdims)
    added_nodes.append(reduce_max_node)
    added_outputs.append(
        helper.make_tensor_value_info(reduce_max_node.output[0],
                                      TensorProto.FLOAT, shape))
    reduced_edges.append(edge_to_reduce)


def augment_graph(model, static):
    '''
    Adds ReduceMin and ReduceMax nodes to all Conv and MatMul nodes in
    model and ensures their outputs are stored as part of the graph output
        parameter model: loaded FP32 ONNX model to quantize
        return: augmented ONNX model
    '''
    shape_inferred = onnx.shape_inference.infer_shapes(model)
    value_infos = {vi.name: vi for vi in shape_inferred.graph.value_info} 
    value_infos.update({ot.name: ot for ot in shape_inferred.graph.output})
    value_infos.update({vi.name: vi for vi in shape_inferred.graph.input})

    added_nodes = []
    added_outputs = []
    reduced_edges = []
    i = 0
    tensors_to_calibrate = set()
    for node in model.graph.node:
        if node.op_type in QUANTIZATION_CANDIDATES or node.op_type in BINARY_OPS_TO_QUANTIZE:
            output_name = node.output[0]
            if can_quantize_name(node, output_name, value_infos):
                tensors_to_calibrate.add(output_name)
            # In dynamic quantization we can just worry about outputs of QUANTIZATION_CANDIDATES
            # In static mode we need the inputs for those as well
            if static:
                for idx, input_name in enumerate(node.input):
                    if can_quantize_name(node, input_name, value_infos, idx) and input_name not in reduced_edges:
                        tensors_to_calibrate.add(input_name)


    for name in tensors_to_calibrate:
        create_reduce_nodes(name, added_nodes, added_outputs, reduced_edges, value_infos)

    augmented_model = copy.deepcopy(model)
    augmented_model.graph.node.extend(added_nodes)
    augmented_model.graph.output.extend(added_outputs)
    return augmented_model


# Using augmented outputs to generate inputs to quantize.py
def get_intermediate_outputs(model_path, session, inputs, calib_mode='naive'):
    '''
    Gather intermediate model outputs after running inference
        parameter model_path: path to augmented FP32 ONNX model
        parameter inputs: list of loaded test inputs (or image matrices)
        parameter calib_mode: type 'naive' gives (ReduceMin, ReduceMax) pairs
                                for each augmented node across test data sets, where
                                the first element is a minimum of all ReduceMin values
                                and the second element is a maximum of all ReduceMax
                                values; more techniques can be added based on further experimentation
                                to improve the selection of the min max values. For example: some kind
                                of noise reduction can be applied before taking the min and max values.
        return: dictionary mapping added node names to (ReduceMin, ReduceMax) pairs
    '''
    model = onnx.load(model_path)
    num_model_outputs = len(
        model.graph.output)  # number of outputs in original model
    num_inputs = len(inputs)
    input_names = [i.name for i in session.get_inputs()]
    num_input_names = len(input_names)
    intermediate_outputs = [
        session.run(
            [], {input_names[j]: inputs[i][j]
                 for j in range(num_input_names)}) for i in range(num_inputs)
    ]

    # Creating dictionary with output results from multiple test inputs
    node_output_names = [
        session.get_outputs()[i].name
        for i in range(len(intermediate_outputs[0]))
    ]
    output_dicts = [
        dict(zip(node_output_names, intermediate_outputs[i]))
        for i in range(num_inputs)
    ]
    merged_dict = {}
    for d in output_dicts:
        for k, v in d.items():
            merged_dict.setdefault(k, []).append(v)
    added_node_output_names = node_output_names[num_model_outputs:]
    node_names = [
        added_node_output_names[i].rpartition('_')[0]
        for i in range(0, len(added_node_output_names), 2)
    ]  # output names
    # Characterizing distribution of a node's values across test data sets
    clean_merged_dict = dict(
        (i, merged_dict[i]) for i in merged_dict if i != node_output_names[0])
    if calib_mode == 'naive':
        pairs = [
            tuple([
                float(min(clean_merged_dict[added_node_output_names[i]])),
                float(max(clean_merged_dict[added_node_output_names[i + 1]]))
            ]) for i in range(0, len(added_node_output_names), 2)
        ]
    elif calib_mode == 'mean':
        pairs = [
            tuple([
                float(np.mean(clean_merged_dict[added_node_output_names[i]])),
                float(np.mean(clean_merged_dict[added_node_output_names[i + 1]]))
            ]) for i in range(0, len(added_node_output_names), 2)
        ]
    elif calib_mode == 'median':
        pairs = [
            tuple([
                float(np.median(clean_merged_dict[added_node_output_names[i]])),
                float(np.median(clean_merged_dict[added_node_output_names[i + 1]]))
            ]) for i in range(0, len(added_node_output_names), 2)
        ]
    else:
        raise ValueError(
            'Unknown value for calib_mode. Currently only naive mode is supported.'
        )

    final_dict = dict(zip(node_names, pairs))
    return final_dict


def get_rmin_rmax_for_node(input_name_to_nodes, output_edge_name, quantization_thresholds, rmin, rmax):
    if output_edge_name and output_edge_name in input_name_to_nodes:
        next_nodes = input_name_to_nodes[output_edge_name]
        if len(next_nodes) == 1 and len(next_nodes[0].output) == 1 and next_nodes[0].output[0] in input_name_to_nodes:
            next_next_nodes = input_name_to_nodes[next_nodes[0].output[0]]
            if len(next_next_nodes) == 1:
                if next_nodes[0].op_type == 'Relu' and next_next_nodes[0].op_type == 'MaxPool':
                    print("Found Relu/Maxpool")
                    thresh = quantization_thresholds[next_next_nodes[0].output[0]]
                    return (thresh[0], thresh[1])

            # We update the output range min and max when next node is clip or relu
            # With this technique we can remove these 2 ops and
            # reduce the output range which in turn helps to improve accuracy
            if next_nodes[0].op_type == 'Relu':
                return (max(rmin, 0), rmax)
            if next_nodes[0].op_type == 'Clip':
                clip_min = next_nodes[0].attribute[0].f
                clip_max = next_nodes[0].attribute[1].f
                return (max(rmin, clip_min), min(rmax, clip_max))
        elif len(next_nodes) > 1 and 'Relu' in [x.op_type for x in next_nodes]:
            raise ValueError(
                "Not reducing output range as output also goes to non-Relu nodes: {}"
                .format(next_nodes))

    return (rmin, rmax)


def calculate_scale_zeropoint(rmin, rmax, mode):
    zp_and_scale = []
    # adjust rmin and rmax such that 0 is included in the range. This is required
    # to make sure zero can be uniquely represented.
    rmin = min(rmin, 0)
    rmax = max(rmax, 0)

    if mode == 'int8':
        max_range = max(abs(rmin), abs(rmax))
        scale = (np.float32(max_range)) / 127 if not math.isclose(max_range, 0, abs_tol=1e-8) else np.float32(1)
        zero_point = np.int8(0)
    else:
        scale = np.float32((rmax - rmin) / 255 if not math.isclose(rmin, rmax, abs_tol=1e-8) else np.float32(1))
        initial_zero_point = (0 - rmin) / scale
        zero_point = np.uint8(round(max(0, min(255, initial_zero_point))))

    zp_and_scale.append(zero_point)
    zp_and_scale.append(scale)
    return zp_and_scale


def calculate_quantization_params(model, quantization_thresholds, static,
                                  mode):
    '''
        Given a model and quantization thresholds, calculates the quantization params.
    :param model: ModelProto to quantize
    :param quantization_thresholds:
        Dictionary specifying the min and max values for outputs of conv and matmul nodes.
        The quantization_thresholds should be specified in the following format:
            {
                "param_name": [min, max]
            }
        example:
            {
                'Conv_3:0': [np.float32(0), np.float32(0.5)],
                'Conv_4:0': [np.float32(1), np.float32(3.5)]
            }
    :return: Dictionary containing the zero point and scale values for outputs of conv and matmul nodes.
        The dictionary format is
            {
                "param_name": [zero_point, scale]
            }
    '''
    if quantization_thresholds is None:
        raise ValueError(
            'quantization thresholds is required to calculate quantization params (zero point and scale)'
        )

    quantization_params = {}
    input_name_to_nodes = get_input_name_to_nodes(model)

    for index, node in enumerate(model.graph.node):
        node_output_name = node.output[0]
        if node_output_name in quantization_thresholds:
            node_thresholds = quantization_thresholds[node_output_name]
            node_params = calculate_scale_zeropoint(
                *get_rmin_rmax_for_node(input_name_to_nodes, node_output_name, quantization_thresholds, node_thresholds[0], node_thresholds[1]),
                mode)
            quantization_params[node_output_name] = node_params
        if static:
            for idx, node_input_name in enumerate(node.input):
                if node_input_name in quantization_thresholds:
                    if node_input_name not in quantization_params:
                        node_thresholds = quantization_thresholds[
                            node_input_name]
                        node_params = calculate_scale_zeropoint(
                            *get_rmin_rmax_for_node(input_name_to_nodes, None, quantization_thresholds, node_thresholds[0], node_thresholds[1]),
                            mode)
                        quantization_params[node_input_name] = node_params

    return quantization_params


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
        return [
            load_single_test_data(test_data_dir, num_expected_inputs,
                                  preprocess_method)
        ]

def main():
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(
        description='parsing model and test data set paths')
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
    model_path = args.model_path
    output_model_path = args.output_model_path
    images_folder = args.dataset_path
    calib_mode = "naive"
    size_limit = args.dataset_size

    # Generating augmented ONNX model
    augmented_model_path = 'augmented_model.onnx'
    model = onnx.load(model_path)
    augmented_model = augment_graph(model, static=args.static)
    onnx.save(augmented_model, augmented_model_path)

    # Conducting inference
    session = onnxruntime.InferenceSession(augmented_model_path, None)
    num_expected_inputs = len(session.get_inputs())

    # Get input samples for quantization
    # If the input folder points to a bunch of protos, use that
    if not len(glob.glob(os.path.join(images_folder, '*.jpg'))):
        inputs = load_test_data(images_folder, num_expected_inputs, size_limit,
                                args.data_preprocess)
    else:
        # NOTE: This is currently broken because I changed the format of inputs
        (samples, channels, height, width) = session.get_inputs()[0].shape
        inputs = load_batch(images_folder, height, width, size_limit,
                            args.data_preprocess)

    print('Num cases {}, num inputs for each cases {}'.format(
        len(inputs), len(inputs[0])))
    dict_for_quantization = get_intermediate_outputs(model_path, session,
                                                     inputs, calib_mode)
    print(dict_for_quantization)
    quantization_params_dict = calculate_quantization_params(
        model,
        quantization_thresholds=dict_for_quantization,
        static=args.static,
        mode=args.mode)
    print(quantization_params_dict)
    calibrated_quantized_model = quantize(
        onnx.load(model_path),
        quantization_mode=QuantizationMode.QLinearOps,
        quantization_params=quantization_params_dict,
        static=args.static,
        symmetric_activation=args.mode == 'int8',
        symmetric_weight=args.mode == 'int8')
    onnx.save(calibrated_quantized_model, output_model_path)

    print("Calibrated, quantized model saved.")


if __name__ == '__main__':
    main()
