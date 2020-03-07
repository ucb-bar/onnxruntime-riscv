# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import onnx
import onnx.numpy_helper
import struct

import numpy as np
from onnx import onnx_pb as onnx_proto

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
onnx_op_set_version = 11

type_to_name = {
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
}

# Quantization mode
# IntegerOps: Use IntegerOps in quantized model. Only ConvInteger and MatMulInteger ops are supported now.
# QLinearOps: Use QLinearOps in quantized model. Only QLinearConv and QLinearMatMul ops are supported now.
class QuantizationMode():
    IntegerOps = 0
    QLinearOps = 1

class TensorLayout():
    NCHW = 0
    NHWC = 1

quantization_modes = [getattr(QuantizationMode, attr) for attr in dir(QuantizationMode)\
    if not callable(getattr(QuantizationMode, attr)) and not attr.startswith("__")]

class QuantizedInitializer:
    '''
        Represents a linearly quantized weight input from ONNX operators
    '''
    def __init__(self, name, initializer, rmins, rmaxs, zero_points, scales, data=[], quantized_data=[], axis=None,
                 qType=onnx_proto.TensorProto.UINT8):
        self.name = name
        self.initializer = initializer  # TensorProto initializer in ONNX graph
        self.rmins = rmins  # List of minimum range for each axis
        self.rmaxs = rmaxs  # List of maximum range for each axis
        self.zero_points = zero_points  # 1D tensor of zero points computed for each axis. scalar if axis is empty
        self.scales = scales  # 1D tensor of scales computed for each axis. scalar if axis is empty
        self.data = data  # original data from initializer TensorProto
        self.quantized_data = quantized_data  # weight-packed data from data
        self.axis = axis  # Scalar to specify which dimension in the initializer to weight pack.
                          # If empty, single zero point and scales computed from a single rmin and rmax
        self.qType = qType # type of quantized data.

class QuantizedValueType():
    Input = 0
    Initializer = 1

class QuantizedValue:
    '''
    Represents a linearly quantized value (input\output\intializer)
    '''
    def __init__(self, name, new_quantized_name, scale_name, zero_point_name, quantized_value_type, axis=None,
                 qType=onnx_proto.TensorProto.UINT8, tensor_layout=TensorLayout.NCHW):
        self.original_name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.value_type = quantized_value_type
        self.axis = axis
        self.qType = qType
        self.tensor_layout = tensor_layout

def quantize_data(data, quantize_range, qType):
    '''
        :parameter quantize_range: list of data to weight pack.
        :parameter qType: data type to quantize to. Supported types UINT8 and INT8
        :return: minimum, maximum, zero point, scale, and quantized weights

        To pack weights, we compute a linear transformation
            - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
            - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
                m = max(abs(rmin), abs(rmax))

        and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation
        r = S(q-z), where
            r: real original value
            q: quantized value
            S: scale
            z: zero point
    '''
    rmin = min(min(data), 0)
    rmax = max(max(data), 0)

    if qType == onnx_proto.TensorProto.INT8:
        max_range = max(abs(rmin), abs(rmax))
        scale = (float(max_range)*2) / quantize_range
        zero_point = 0
        quantized_data = (np.asarray(data) / scale).round().astype('b') #signed byte type
    elif qType == onnx_proto.TensorProto.UINT8:
        scale = (float(rmax) - rmin) / quantize_range if rmin != rmax else 1
        zero_point = round((0 - rmin) / scale) # round to nearest integer
        quantized_data = ((np.asarray(data) / scale).round() + zero_point).astype('B') # unsigned byte type
    else:
        raise ValueError("Unexpected data type {} requested. Only INT8 and UINT8 are supported.")

    return rmin, rmax, zero_point, scale, quantized_data


def _attribute_to_kwarg(attribute):
    '''
    Convert attribute to kwarg format for use with onnx.helper.make_node.
        :parameter attribute: attribute in AttributeProto format.
        :return: attribute in {key: value} format.
    '''
    if (attribute.type == 0):
        raise ValueError('attribute {} does not have type specified.'.format(attribute.name))

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    if (attribute.type == 1):
        value = attribute.f
    elif (attribute.type == 2):
        value = attribute.i
    elif (attribute.type == 3):
        value = attribute.s
    elif (attribute.type == 4):
        value = attribute.t
    elif (attribute.type == 5):
        value = attribute.g
    elif (attribute.type == 6):
        value = attribute.floats
    elif (attribute.type == 7):
        value = attribute.ints
    elif (attribute.type == 8):
        value = attribute.strings
    elif (attribute.type == 9):
        value = attribute.tensors
    elif (attribute.type == 10):
        value = attribute.graphs
    else:
        raise ValueError('attribute {} has unsupported type {}.'.format(attribute.name, attribute.type))

    return {attribute.name: value}

def _find_by_name(item_name, item_list):
    '''
    Helper function to find item by name in a list.
        parameter item_name: name of the item.
        parameter item_list: list of items.
        return: item if found. None otherwise.
    '''
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None

def _get_mul_node(inputs, output, name):
    '''
    Helper function to create a Mul node.
        parameter inputs: list of input names.
        parameter output: output name.
        parameter name: name of the node.
        return: Mul node in NodeProto format.
    '''
    return onnx.helper.make_node("Mul", inputs, [output], name)

def _find_node_by_name(node_name, graph, new_nodes_list):
    '''
    Helper function to check if a node exists in a graph or
    new set of nodes created during quantization.
        parameter node_name: name of the node.
        parameter graph: GraphProto.
        parameter new_nodes_list: list of nodes added during quantization.
        return: NodeProto if found. None otherwise.
    '''
    graph_nodes_list = list(graph.node) # deep copy
    graph_nodes_list.extend(new_nodes_list)
    node = _find_by_name(node_name, graph_nodes_list)
    return node

def _add_initializer_if_not_present(graph, name, value, shape, type):
    '''
    Helper function to add an initializer if it is not present in the graph.
        parameter graph: GraphProto.
        parameter name: Initializer's name.
        parameter value: Initializer's value.
        parameter shape: Initializer's shape.
        parameter type: Initializer's type.
    '''
    if _find_by_name(name, graph.initializer) is None:
        initializer = onnx.helper.make_tensor(name, type, shape, value)
        value_info = onnx.helper.make_tensor_value_info(name, type, shape)
        graph.initializer.extend([initializer])
        graph.input.extend([value_info])

def _get_qrange_for_qType(qType):
    '''
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    '''
    if qType == onnx_proto.TensorProto.UINT8:
        return 255  # 2^b - 1
    elif qType == onnx_proto.TensorProto.INT8:
        return 254  # [-(2^{b-1}-1), 2^{b-1}-1]: [-127, 127] for 8 bits.
    else:
        raise ValueError('unsupported quantization data type')

def _find_nodes_using_initializer(graph, initializer):
    '''
    Helper function to find all nodes with an initializer as a input.
        parameter graph: GraphProto.
        parameter initializer: Initializer in TensorProto format.
        return: List of nodes.
    '''
    result = []
    for node in graph.node:
        for node_input in node.input:
            if node_input == initializer.name:
                result.append(node)
    return result

class ONNXQuantizer:
    def __init__(self, model, per_channel, mode, static, fuse_dynamic_quant, weight_qType, input_qType,
            quantization_params, nodes_to_quantize, force_nhwc_conv):
        self.model = model
        self.per_channel = per_channel # weight-pack per channel        
        self.mode = mode # QuantizationMode.Value
        self.static = static # use static quantization for inputs.
        self.fuse_dynamic_quant = fuse_dynamic_quant
        self.input_qType = input_qType # quantize input type
        self.weight_qType = weight_qType  # quantize data type
        self.quantization_params = quantization_params
        self.nodes_to_quantize = nodes_to_quantize # specific nodes to quantize
        self.force_nhwc_conv = force_nhwc_conv # Whether to use NHWC format for convolution

        if not self.mode in quantization_modes:
            raise ValueError('unsupported quantization mode {}'.format(self.mode))

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        self.fixed_one_name = "fixed_one"
        # For int8 data-type, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        # List of quantized weights
        self._quantized_weights = []
        # Map of all original value names to quantized value names
        self.quantized_value_map = {}
        # Cached map of NCHW value names to NWHC format value names for convert
        self.nchw_to_nhwc_value = {}
        # Cached map of NWHC value names to NHWC format value names for deconvert
        self.nhwc_to_nchw_value = {}
    
    def quantize_model(self):
        # Create a new topologically sorted list for quantizing a model
        new_list = []
        for node in self.model.graph.node:
            # if a list of ops to be quantized is provided then only quantize those ops
            if self.nodes_to_quantize is not None and node.name not in self.nodes_to_quantize:
                new_list +=self._handle_other_ops(node, new_list)
            # only onnx domain ops can be quantized today
            elif node.domain != "ai.onnx" and node.domain != '':
                new_list +=self._handle_other_ops(node, new_list)
            else:
                if node.op_type == 'Conv':
                    new_list += self._quantize_convolution(node, new_list)
                elif node.op_type == 'MatMul':
                    new_list += self._quantize_matmul(node, new_list)
                elif node.op_type == 'Gather':
                    new_list += self._quantize_gather_ops(node, new_list)
                elif node.op_type == 'Relu' or node.op_type == 'Clip':
                    new_list +=self._handle_activation_ops(node, new_list)
                else:
                    new_list +=self._handle_other_ops(node, new_list)

        # extend is used to append to the list for a protobuf fields
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
        self.model.graph.ClearField('node')
        self.model.graph.node.extend(new_list)

        # Remove weights which are already quantized from graph.
        self._remove_quantized_weights()

        # update opset. 
        opset_info = next((opset for opset in self.model.opset_import if opset.domain == '' or opset.domain == onnx_domain), None)
        if opset_info is not None:
            self.model.opset_import.remove(opset_info)
        self.model.opset_import.extend([onnx.helper.make_opsetid(onnx_domain, onnx_op_set_version)])

        return self.model

    def find_weight_data(self, initializer):
        '''
            :param initializer: TensorProto initializer object from a graph
            :return: a list of initialized data in a given initializer object
        '''
        if initializer.data_type == onnx_proto.TensorProto.FLOAT:
            weights = onnx.numpy_helper.to_array(initializer)
        else:
            raise ValueError('Model contains conv operator weights in {}. Only float type quantization is supported.'.format(
                type_to_name[initializer.data_type]))
        return weights

    def _remove_quantized_weights(self):
        ''' Remove the weights which are already quantized from graph initializer list.
            This function assumes that after quantization, all nodes that previously use a weight:
                - use output from DequantizeLinear as input if they do not support quantization.
                - use quantized weight if they support quantization.
        '''
        for weight in self._quantized_weights:
            # Remove existing weight initializer
            self.model.graph.initializer.remove(weight.initializer)

            # Removing input weight to a convolution
            try:
                weight_input = next(val for val in self.model.graph.input if val.name == weight.name)
                self.model.graph.input.remove(weight_input)
            except StopIteration:
                if self.model.ir_version < 4:
                    raise ValueError('invalid weight name {} found in the graph (not a graph input) '.format(weight.name))


    def _update_graph(self, weight):
        '''
            Given a weight object, update the graph by doing the following:
             - remove old initializer, update new initializers for quantized weight, zero point, and scale
             - remove old weight input, update with new inputs for quantized weight, zero point, and scale
            This function does NOT update the nodes in the graph, just initializers and inputs
        '''
        quantized_value = self.quantized_value_map[weight.name]
        assert(quantized_value is not None)
        packed_weight_name = quantized_value.q_name
        scale_name = quantized_value.scale_name
        zero_point_name = quantized_value.zp_name

        # Update packed weight, zero point, and scale initializers
        packed_weight_np_data = np.asarray(weight.quantized_data,
            dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight.qType]).reshape(weight.initializer.dims)
        packed_weight_initializer = onnx.numpy_helper.from_array(packed_weight_np_data, packed_weight_name)

        if weight.axis is not None:
            zero_scale_shape = [weight.initializer.dims[weight.axis]]
        else: # scale and zero point must be scalar
            zero_scale_shape = []
        zero_point_type = weight.qType
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape, weight.scales)
        zero_initializer = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_scale_shape, weight.zero_points)

        self.model.graph.initializer.extend([packed_weight_initializer, scale_initializer, zero_initializer])

        # Create input for initialized scale and zeros
        packed_weight_value_info = onnx.helper.make_tensor_value_info(packed_weight_name, weight.qType,
                                        weight.initializer.dims)
        scale_value_info = onnx.helper.make_tensor_value_info(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape)
        zero_point_value_info = onnx.helper.make_tensor_value_info(zero_point_name,
            zero_point_type, zero_scale_shape) # zero_point is int for dequantize operator

        self.model.graph.input.extend([packed_weight_value_info, scale_value_info, zero_point_value_info])

        self._quantized_weights.append(weight)

    def _get_quantized_weight(self, initializer, qType, wanted_layout):
        '''
            :param initializer: TensorProto initializer
            :param qType: type to quantize to
            :return: Weight class with quantization information
        '''
        weights_data = self.find_weight_data(initializer)
        if wanted_layout == TensorLayout.NHWC:
            weights_data = np.transpose(weights_data, (0, 2, 3, 1))
            origin_initializer_dims = initializer.dims
            initializer.dims[:] = [origin_initializer_dims[0], origin_initializer_dims[2], origin_initializer_dims[3], origin_initializer_dims[1]]
        rmin, rmax, zero_point, scale, quantized_weights_data = quantize_data(weights_data.flatten().tolist(),
            _get_qrange_for_qType(qType), qType)

        weight = QuantizedInitializer(initializer.name, initializer, [rmin], [rmax], [zero_point], [scale],
                        weights_data, quantized_weights_data, axis=None, qType=qType)

        # Log entry for this quantized weight
        assert(weight.name not in self.quantized_value_map)
        quantized_value = QuantizedValue(weight.name, weight.name + "_quantized", weight.name + "_scale", weight.name + "_zero_point", QuantizedValueType.Initializer, None, qType, tensor_layout=wanted_layout)
        self.quantized_value_map[weight.name] = quantized_value

        return weight

    def _get_quantized_weight_convolution(self, initializer, qType, wanted_layout):
        '''
            :param initializer: initializer TypeProto to quantize
            :param qType: type to quantize to
            :return: Weight class object with quantization information for a given initializer
        '''
        if not self.per_channel:
            return self._get_quantized_weight(initializer, qType, wanted_layout)

        weights = self.find_weight_data(initializer)
        # Quantize per output channel
        # Assuming (M x C/group x kH x kW) format where M is number of output channels.
        channel_count = initializer.dims[0]
        np_data = np.reshape(weights, initializer.dims)
        rmin_list = []
        rmax_list = []
        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = []
        for i in range(channel_count):
            # for each channel, compute quantization data. Assuming (M x C/group x kH x kW)
            per_channel_data = np_data[i,:,:,:].flatten()
            rmin, rmax, zero_point, scale, quantized_per_channel_data = quantize_data(per_channel_data.flatten().tolist(),
                _get_qrange_for_qType(qType), qType)
            rmin_list.append(rmin)
            rmax_list.append(rmax)
            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(quantized_per_channel_data)
        channel_index = 0 # (M x C/group x kH x kW)
        # combine per_channel_data into one
        reshape_dims = list(initializer.dims)  # deep copy
        reshape_dims[channel_index] = 1  # only one per channel for reshape
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
        for i in range(1, len(quantized_per_channel_data_list)):
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
            quantized_weights = np.concatenate((quantized_weights, channel_weights), axis=0)

        if wanted_layout == TensorLayout.NHWC:
            quantized_weights = np.permute(quantized_weights, (0, 2, 3, 1))
            origin_initializer_dims = initializer.dims
            initializer.dims[:] = [origin_initializer_dims[0], origin_initializer_dims[2], origin_initializer_dims[3], origin_initializer_dims[1]]
        weight = QuantizedInitializer(initializer.name, initializer, rmin_list, rmax_list, zero_point_list,
                        scale_list, weights, quantized_weights.flatten().tolist(), channel_index, qType)
        
        # Make entry for this quantized weight
        assert(weight.name not in self.quantized_value_map)
        quantized_value = QuantizedValue(weight.name, weight.name + "_quantized", weight.name + "_scale", weight.name + "_zero_point", QuantizedValueType.Initializer, None, qType, tensor_layout=wanted_layout)
        self.quantized_value_map[weight.name] = quantized_value

        return weight

    def _get_dynamic_input_quantization_params(self, input_name, nodes_list, qType):
        '''
        Create nodes for dynamic quantization of input and add them to nodes_list.
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            parameter qType: type to quantize to.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        if qType == onnx_proto.TensorProto.INT8:
            return self._get_dynamic_input_quantization_params_int8(input_name, nodes_list)

        return self._get_dynamic_input_quantization_params_uint8(input_name, nodes_list)

    def _get_dynamic_input_quantization_params_int8(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input to nit8 and add them to nodes_list        
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        qType = onnx_proto.TensorProto.INT8

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name],
            [reduce_min_name + ":0"], reduce_min_name, keepdims=0)
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name],
            [reduce_max_name + ":0"], reduce_max_name, keepdims=0)
        nodes_list.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node("Abs", [reduce_min_node.output[0]],
            [reduce_min_abs_name + ":0"], reduce_min_abs_name)
        nodes_list.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node("Abs", [reduce_max_node.output[0]],
            [reduce_max_abs_name + ":0"], reduce_max_abs_name)
        nodes_list.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node("Max", [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
            [abs_max_name + ":0"], abs_max_name)
        nodes_list.append(abs_max_node)

        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        _add_initializer_if_not_present(self.model.graph, self.fixed_qrange_int8_name,
            [_get_qrange_for_qType(qType)/2.0], [], onnx_proto.TensorProto.FLOAT)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node("Div", [abs_max_node.output[0], self.fixed_qrange_int8_name],
            [input_scale_name], scale_div_name)
        nodes_list.append(scale_div_node)

        # Zero point
        _add_initializer_if_not_present(self.model.graph, self.fixed_zero_zp_name,
            [0], [], qType)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_uint8(self, input_name, nodes_list):
        '''
        Create nodes for dynamic quantization of input to uint8 and add them to nodes_list
            parameter input_name: Name of the input.
            parameter nodes_list: new nodes are appended to this list.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''
        qType = onnx_proto.TensorProto.UINT8
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node("ReduceMin", [input_name],
            [reduce_min_name + ":0"], reduce_min_name, keepdims=0)
        nodes_list.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node("ReduceMax", [input_name],
            [reduce_max_name + ":0"], reduce_max_name, keepdims=0)
        nodes_list.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        _add_initializer_if_not_present(self.model.graph, self.fixed_qrange_uint8_name,
            [_get_qrange_for_qType(qType)], [], onnx_proto.TensorProto.FLOAT)
        _add_initializer_if_not_present(self.model.graph, self.fixed_zero_name,
            [0.0], [], onnx_proto.TensorProto.FLOAT)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node("Sub", [reduce_max_node.output[0], reduce_min_node.output[0]],
            [scale_sub_name + ":0"], scale_sub_name)
        nodes_list.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node("Div", [scale_sub_node.output[0], self.fixed_qrange_uint8_name],
            [input_scale_name], scale_div_name)
        nodes_list.append(scale_div_node)

        # Compute zero point
        #   Subtract zero and rmin
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node("Sub", [self.fixed_zero_name, reduce_min_node.output[0]],
            [zp_sub_name + ":0"], zp_sub_name)
        nodes_list.append(zp_sub_node)
        #   Divide by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node("Div", [zp_sub_node.output[0], input_scale_name],
            [zp_div_name + ":0"], zp_div_name)
        nodes_list.append(zp_div_node)
        #   Compute floor
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output,
            [zp_floor_name + ":0"], zp_floor_name)
        nodes_list.append(zp_floor_node)
        #   Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output,
            [input_zp_name], zp_cast_name, to=qType)
        nodes_list.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []

    def _get_quantization_params(self, param_name):
        '''
        Create initializers and inputs in the graph for zero point and scale of output.
        Zero point and scale values are obtained from self.quantization_params if specified.

            parameter output_name: Name of the output.
            return: scale_name, zero_point_name, scale_shape, zero_point_shape.
        '''        
        if self.quantization_params is None or param_name not in self.quantization_params:
            return False, "", "", "", ""
        params = self.quantization_params[param_name]
        if params is None or len(params) != 2:
            raise ValueError("Quantization parameters should contain zero point and scale. "
                "Specified values for output {}: {}".format(output_name, params))

        if not np.isscalar(params[0]):
            raise ValueError("Zero point for output {} should be a scalar value. Value specified: {}".format(
                output_name, params[0]))
        if not np.isscalar(params[1]):
            raise ValueError("Scale for output {} should be a scalar value. Value specified: {}".format(
                output_name, params[1]))

        zero_point_values = [params[0].item()]
        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        zero_point_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[params[0].dtype]

        scale_values = [params[1].item()]
        scale_shape = []
        scale_name = param_name + "_scale"

        # Add initializers
        _add_initializer_if_not_present(self.model.graph, zero_point_name, zero_point_values, zero_point_shape,
            zero_point_type)
        _add_initializer_if_not_present(self.model.graph, scale_name, scale_values, scale_shape,
            onnx_proto.TensorProto.FLOAT)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantize_input_nodes(self, node, input_index, qType):
        '''
        Given a input for a node (which is not a initializer), this function
            - add elements to graph to compute zero point and scale for this input.
            - add new QuantizeLinear nodes to quantize the input.

            parameter node: node being quantized in NodeProto format.
            parameter input_index: index of input in node.input.
            parameter qType: type to quantize to.
            return: List of newly created nodes in NodeProto format.
        '''
        input_name = node.input[input_index]
        output_name = input_name + "_quantized"

        data_found, scale_name, zp_name, scale_shape, zp_shape = \
                self._get_quantization_params(input_name)

        if self.static:
            if data_found == False:
                raise ValueError("Quantization parameters are not specified for param {}."
                "In static mode quantization params for inputs and outputs of odes to be quantized are required.".format(input_name))

            qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name], 
                [output_name], input_name + "_QuantizeLinear")
            
            return [qlinear_node]
            
        else:
            if data_found == True:
                qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name], 
                    [output_name], input_name + "_QuantizeLinear")
                return [qlinear_node]
            else:
                # Scale and Zero Points not available for this input. Add nodes to dynamically compute it
                if self.fuse_dynamic_quant and qType == onnx_proto.TensorProto.UINT8:
                    scale_name = input_name + "_scale"
                    zeropoint_name = input_name + "_zero_point"
                    qlinear_node = onnx.helper.make_node("DynamicQuantizeLinear", [input_name],
                        [output_name, scale_name, zeropoint_name], input_name + "_QuantizeLinear")
                    return [qlinear_node]
                
                else:
                    nodes = []
                    scale_name, zp_name, scale_shape, zp_shape = \
                        self._get_dynamic_input_quantization_params(input_name, nodes, qType)
                    qlinear_node = onnx.helper.make_node("QuantizeLinear", [input_name, scale_name, zp_name], 
                        [output_name], input_name + "_QuantizeLinear")
            
                    return nodes + [qlinear_node]           

    def _get_bias_add_nodes(self, nodes, node, last_output, quantized_bias_name):
        '''
        Given a node, this function handles bias add by adding a "reshape" node on bias and an "add" node

            parameter nodes: new nodes would be appended into nodes
            parameter node: current node (Conv)
            parameter last_output: output of previous node (input to bias add)
            return: the name of output
        '''
        # Add an Add operation for bias
        # Add reshape for correct broadcase
        reshape_input = [quantized_bias_name]

        # Add tensors for the shape to be reshaped to
        _add_initializer_if_not_present(self.model.graph, "reshape_shape",
                                        [1,-1,1,1], [4], onnx_proto.TensorProto.INT64)
        reshape_input.append('reshape_shape')
        reshape_op_output = node.output[0] + "_reshape"
        reshape_node = onnx.helper.make_node("Reshape", reshape_input, [reshape_op_output],
                                            quantized_bias_name+"reshape")
        nodes.append(reshape_node)

        bias_add_input = [last_output]
        bias_add_input.append(reshape_op_output)
        add_node_output = node.output[0] + "_bias_add"
        add_node = onnx.helper.make_node("Add", bias_add_input, [add_node_output],
                                        quantized_bias_name + "bias_add")
        nodes.append(add_node)
        return add_node_output

    def _update_unsupported_nodes_using_weight(self, weight, new_nodes_list, tensor_layout):        
        '''Find all nodes using a weight that do not support quantization and
        add a DequantizeLinear node before those nodes. This includes all nodes except Conv, MatMul.

            parameter weight: Weight object
            parameter new_nodes_list: List of new nodes created before processing current node.
            return: List of new nodes created.
        '''
        nodes_using_weight = _find_nodes_using_initializer(self.model.graph, weight.initializer)
        unsupported_nodes = [node for node in nodes_using_weight if node.op_type not in ["Conv", "MatMul", "Gather"]]

        nodes_list = []
        dequantize_linear_name = weight.name + "_DequantizeLinear"
        output_name = weight.name + "_dequantized"

        # Check if DequantizeLinear node needs to be added to graph.
        if len(unsupported_nodes) != 0 and \
            _find_node_by_name(dequantize_linear_name, self.model.graph, new_nodes_list) is None:
            inputs = [weight.name + "_quantized", weight.name + "_scale", weight.name + "_zero_point"]
            node = onnx.helper.make_node("DequantizeLinear", inputs, [output_name],
                                         dequantize_linear_name)
            nodes_list.append(node)

        # If we need to convert back to NCHW tensor_layout
        if tensor_layout != TensorLayout.NCHW:
            input_name = output_name
            node_name = output_name + "__to_nchw"
            output_name = output_name + "_nchw"
            transpose_node = onnx.helper.make_node("Transpose", [input_name], [output_name], perm=[0, 3, 1, 2] , name=node_name)

        # Update unsupported nodes to take dequantized weight as input.
        for node in unsupported_nodes:
            for i, node_input in enumerate(node.input):
                if node_input == weight.name:
                    node.input[i] = output_name

        return nodes_list

    def _dynamic_quantize_bias(self, input_name, weight_scale_name, bias_name, quantized_bias_name, new_node_list):
        '''
        Adds series of nodes required to quantize the bias dynamically.
            parameter input_name: Input name
            parameter weight_scale_name: Weight scale.
            parameter bias_scale_name: Bias to quantize.
            parameter quantied_bias_name: Output name to use for quantized bias.
        '''
        qType = onnx_proto.TensorProto.INT32
        
        input_scale_name = input_name + "_scale"
        bias_scale_node = onnx.helper.make_node("Mul", [input_scale_name, weight_scale_name], [bias_name + "_scale"], bias_name + "_scale_node")
        new_node_list.append(bias_scale_node)

        quantize_bias_node = onnx.helper.make_node("Div", [bias_name, bias_scale_node.output[0]],
            [bias_name + "_tmp_quant:0"], bias_name + "_tmp_qaunt")
        new_node_list.append(quantize_bias_node)

        bias_rounded_node = onnx.helper.make_node("Floor", quantize_bias_node.output,
            [bias_name + "_quant_rounded:0"], bias_name + "_quant_rounded")
        new_node_list.append(bias_rounded_node)
        
        bias_cast_node = onnx.helper.make_node("Cast", bias_rounded_node.output,
            [quantized_bias_name], quantized_bias_name + "_node", to=qType)
        new_node_list.append(bias_cast_node)
        
        return 


    def _quantize_bias(self, node, new_node_list):
        '''
        Quantized the bias. Zero Point == 0 and Scale == Input_Scale * Weight_Scale 
        '''

         # get scale for weight 
        weight_scale_name = self.quantized_value_map[node.input[1]].scale_name
        weight_initializer = _find_by_name(weight_scale_name, self.model.graph.initializer)
        weight_scale = self.find_weight_data(weight_initializer)  

        # get bias
        bias_name = node.input[2]
        bias_initializer = _find_by_name(bias_name, self.model.graph.initializer)
        bias_data = self.find_weight_data(bias_initializer)
        quantized_bias_name = bias_name + "_quantized"      

        # input scale is not provided and this input is dynamically quantized so it is not pre-computed at this point
        # so resort to dynamic quantization for bias
        if self.quantization_params is None or node.input[0] not in self.quantization_params and node.input[0] not in self.quantized_value_map:
            self._dynamic_quantize_bias(node.input[0], weight_scale_name, bias_name, quantized_bias_name, new_node_list)
        else:
            # get scale for input
            if node.input[0] in self.quantized_value_map:
                input_scale_name = self.quantized_value_map[node.input[0]].scale_name
            elif node.input[0] in self.quantization_params:
                _, input_scale_name, _, _, _ = self._get_quantization_params(node.input[0])
            else:
                raise ValueError("Expected {} to be in quantized value map for static quantization".format(node.input[0]))

            inputscale_initializer = _find_by_name(input_scale_name, self.model.graph.initializer)
            input_scale = self.find_weight_data(inputscale_initializer)
            # calcuate scale for bias
            bias_scale_name = node.input[2] + "_scale"
            bias_scale = input_scale * weight_scale
            # print(bias_scale)
     
            # quantize bias
            quantized_data = (np.asarray(bias_data) / bias_scale).round().astype(np.int32)
            # print(quantized_data)

            #update bias initializer        
            bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
            packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
            self.model.graph.initializer.extend([packed_bias_initializer])

            bias_value_info = onnx.helper.make_tensor_value_info(quantized_bias_name, onnx_proto.TensorProto.INT32, bias_initializer.dims)
            self.model.graph.input.extend([bias_value_info])

            # log entries for this quantized bias value
            quantized_bias_entry = QuantizedInitializer(bias_name, bias_initializer, [0], [0], [0], [bias_scale],
                            bias_data, quantized_data, qType=onnx_proto.TensorProto.INT32)
            self._quantized_weights.append(quantized_bias_entry)
        
            assert(bias_name not in self.quantized_value_map)
            quantized_value = QuantizedValue(bias_name, quantized_bias_name, "", "", QuantizedValueType.Initializer, None, onnx_proto.TensorProto.INT32)
            self.quantized_value_map[bias_name] = quantized_value

        return quantized_bias_name

    def _get_tensor_layout(self, input_name):
        '''
        For an original node input, determines the tensor layout of the quantized version
        '''
        if input_name in self.quantized_value_map:
            return self.quantized_value_map[input_name].tensor_layout
        return TensorLayout.NCHW

    def _convert_to_nhwc(self, quantized_input_name):
        '''
        Given a quantized input param, output a quantized param corresponding to a NHWC transpose'd version
        '''
        # If we've already computed this before then use cached
        if quantized_input_name in self.nchw_to_nhwc_value:
            return (self.nchw_to_nhwc_value[quantized_input_name], [])
        new_out = quantized_input_name + "_nhwc"
        node_name = quantized_input_name + "__to_nhwc"
        #Convert NCHW to NHWC
        transpose_node = onnx.helper.make_node("Transpose", [quantized_input_name], [new_out], perm=[0, 2, 3, 1] , name=node_name)
        self.nchw_to_nhwc_value[quantized_input_name] = new_out

        return (new_out, [transpose_node])
        
    def _convert_to_nchw(self, input_name):
        '''
        Given a quantized input param, output a quantized param corresponding to a NCHW transpose'd version
        '''
        if input_name in self.nhwc_to_nchw_value:
            return self.nhwc_to_nchw_value[input_name]

        new_out = input_name + "_nchw"
        node_name = input_name + "__to_nchw"
        #Convert NHWC to NCHW
        transpose_node = onnx.helper.make_node("Transpose", [input_name], [new_out], perm=[0, 3, 1, 2] , name=node_name)
        self.nhwc_to_nchw_value[input_name] = new_out
        return (new_out, [transpose_node])

    def _quantize_inputs(self, node, indices, new_nodes_list, wanted_weight_tensor_layout=TensorLayout.NCHW):
        '''
        Given a node, this function quantizes the inputs as follows:
            - If input is a initializer, quantize the initializer data, replace old initializer
              with new initializer
            - Else, add QuantizeLinear nodes to perform quantization

            parameter node: node being quantized in NodeProto format.
            parameter indices: input indices to quantize.
            parameter new_nodes_list: List of new nodes created before processing this node. This is used to
                                      check that two QuantizeLinear nodes are not being added for same input.
            
            parameter wanted_weight_tensor_layout: What layout format to use if we're quantizing a weight

            return: (List of quantized input names,
                     List of zero point names used for input quantization,
                     List of scale names used for input quantization,
                     List of new QuantizeLinear nodes created)
        '''
        assert (node.op_type == "Conv" or node.op_type == "MatMul" or node.op_type == "Gather" or node.op_type == "Relu")

        quantized_input_names = []
        zero_point_names = []
        scale_names = []
        nodes = []

        for input_index in indices:
            node_input = node.input[input_index]

            # Find if this input is already quantized
            if node_input in self.quantized_value_map:
                quantized_value = self.quantized_value_map[node_input]
                qType = self.weight_qType if quantized_value.value_type == QuantizedValueType.Initializer else self.input_qType
                if quantized_value.qType != qType: 
                    #print(node_input)
                    #print(quantized_value.qType)
                    #print(qType)
                    raise ValueError("{} is being used by multiple nodes which are being quantized to different types. "
                "This is not suported.", node_input)

                quantized_input_names.append(quantized_value.q_name)
                scale_names.append(quantized_value.scale_name)
                zero_point_names.append(quantized_value.zp_name)
                continue

            # Quantize the input
            initializer = _find_by_name(node_input, self.model.graph.initializer)
            if initializer is not None:
                if node.op_type == "Conv":
                    weight = self._get_quantized_weight_convolution(initializer, self.weight_qType, wanted_weight_tensor_layout)
                else:
                    weight = self._get_quantized_weight(initializer, self.weight_qType, wanted_weight_tensor_layout)

                # Update graph
                nodes.extend(self._update_unsupported_nodes_using_weight(weight, new_nodes_list, wanted_weight_tensor_layout))
                self._update_graph(weight)

                quantized_input_names.append(weight.name + "_quantized")
                zero_point_names.append(weight.name + "_zero_point")
                scale_names.append(weight.name + "_scale")
            else:
                # Add QuantizeLinear node.
                qlinear_node = _find_node_by_name(node_input + "_QuantizeLinear", self.model.graph, new_nodes_list)
                if qlinear_node is None:
                    quantize_input_nodes = self._get_quantize_input_nodes(node, input_index, self.input_qType)
                    nodes.extend(quantize_input_nodes)
                    qlinear_node = quantize_input_nodes[-1]

                if qlinear_node.op_type == "QuantizeLinear":
                    quantized_input_names.extend(qlinear_node.output)
                    scale_names.append(qlinear_node.input[1])
                    zero_point_names.append(qlinear_node.input[2])
                else:
                    quantized_input_names.append(qlinear_node.output[0])
                    scale_names.append(qlinear_node.output[1])
                    zero_point_names.append(qlinear_node.output[2])


        return (quantized_input_names, zero_point_names, scale_names, nodes)
 
    def _handle_other_ops(self, node, new_nodes_list):
        '''
        Given a node which does not support quantization(Conv, Matmul, Gather), this method 
        checks whether the input to this node is quantized and adds a DequantizeLinear node 
        to dequantize this input back to FP32

            parameter node: Current node
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of new nodes created
        '''
        nodes = []
        for index, node_input in enumerate(node.input):
            if node_input in self.quantized_value_map:
                node_input_altered = True
                input_name = node.input[index]
                quantized_value = self.quantized_value_map[input_name]
                quantized_input_name = quantized_value.q_name

                if quantized_value.tensor_layout != TensorLayout.NCHW:
                    (nchw_input, transpose) = self._convert_to_nchw(quantized_input_name)
                    quantized_input_name = nchw_input
                    nodes.extend(transpose)

                # Add DequantizeLinear Node for this input
                dqlinear_name = input_name + "_DequantizeLinear"
                dqlinear_node = _find_node_by_name(dqlinear_name, self.model.graph, new_nodes_list)
                if dqlinear_node is None:
                    dqlinear_inputs = [quantized_input_name, quantized_value.scale_name, quantized_value.zp_name]
                    dequantize_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [input_name], dqlinear_name)
                    nodes.append(dequantize_node)

                else:
                    # DQ op is already present, assert it's output matches the input of current node
                    assert(input_name == dqlinear_node.output[0])

        # Append the original node
        nodes.append(node)
        return nodes

    def _handle_activation_ops(self, node, new_node_list):
        '''
        Checks whether the give activation op can be removed from the graph. When mode is QLinearOps, 
        the output quatization params are calculated based on outputs from activation nodes, 
        therefore these nodes can be removed from the graph if they follow a quantized op.
        
            parameter node: Current node
            parameter new_nodes_list: List of new nodes created before processing current node
            return: List of nodes
        '''
        assert(node.op_type == "Relu" or node.op_type == 'Clip')
        if self.mode is not QuantizationMode.QLinearOps:
            return [node]
        # When mode is QLinearOps, the output quatization params are calculated based on outputs from
        # activation nodes, therefore these nodes can be removed from the graph if they follow a quantized op.
        # If input to this node is not quantized then keep this node
        if node.input[0] not in self.quantized_value_map:
            return [node]

        if self.input_qType == onnx_proto.TensorProto.UINT8:
            quantized_value = self.quantized_value_map[node.input[0]]
            self.quantized_value_map[node.output[0]] = quantized_value
            return []
        elif node.op_type == "Relu" and self.input_qType == onnx_proto.TensorProto.INT8:
            (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0], new_node_list)
            q_input = self.quantized_value_map[node.input[0]]
            new_out = node.output[0] + "_quantized"
            qlinear_relu = onnx.helper.make_node("QLinearRelu", quantized_input_names, [new_out], node.name)
            q_output = QuantizedValue(node.output[0], new_out, q_input.scale_name, q_input.zp_name, q_input.value_type, qType=q_input.qType, tensor_layout=q_input.tensor_layout)
            self.quantized_value_map[node.output[0]] = q_output
            
            return [qlinear_relu]
        else:
            return self._handle_other_ops(node, new_node_list)

    def _quantize_gather_ops(self, node, new_nodes_list):
        assert (node.op_type == "Gather")
        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0], new_nodes_list)
        
        gather_new_output = node.output[0] + "_quantized"
        quantized_input_name = quantized_input_names[0]

        if self._get_tensor_layout(node.input[0]) != TensorLayout.NCHW:
            (nchw_input, transpose) = self._convert_to_nchw(quantized_input_name)
            quantized_input_name = nchw_input
            nodes.extend(transpose)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], gather_new_output, scale_names[0], zero_point_names[0], QuantizedValueType.Input, qType=self.weight_qType)        
        self.quantized_value_map[node.output[0]] = q_output

        gather_original_output = node.output[0]
        node.output[0] = gather_new_output
        node.input[0] = quantized_input_name
        nodes.append(node)

        return nodes

    def _quantize_convolution_integer_ops(self, node, new_nodes_list):
        '''
        Used when self.mode is QuantizationMode.IntegerOps.
            parameter node: Conv node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "Conv")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], new_nodes_list)

        # quantize bias if exist
        quantized_bias_name = ""
        bias_present = False
        if len(node.input) == 3:
            quantized_bias_name = self._quantize_bias(node, nodes)
            bias_present = True

        conv_integer_output = node.output[0] + "_quantized"
        conv_integer_name = ""
        if node.name != "":
            conv_integer_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(_attribute_to_kwarg(attribute))
        conv_integer_node = onnx.helper.make_node("ConvInteger", quantized_input_names + zero_point_names,
            [conv_integer_output], conv_integer_name, **kwargs)
        nodes.append(conv_integer_node)

        # Add bias add nodes
        if bias_present:
            conv_integer_output = self._get_bias_add_nodes(nodes, node, conv_integer_output, quantized_bias_name)

        # Add cast operation to cast convInteger output to float.
        cast_op_output = conv_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node("Cast", [conv_integer_output], [cast_op_output],
            conv_integer_output + "_cast", to=onnx_proto.TensorProto.FLOAT)
        nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        assert (len(scale_names) == 2)
        if conv_integer_name != "":
            scales_mul_op = conv_integer_name + "_scales_mul"
        else:
            scales_mul_op = scale_names[0] + "_" + scale_names[1] + "_mul"

        scales_mul_node = _find_node_by_name(scales_mul_op, self.model.graph, new_nodes_list)
        if scales_mul_node is None:
            scales_mul_node = _get_mul_node(scale_names, scales_mul_op + ":0", scales_mul_op)
            nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        # Add mul operation to multiply mul_scales_op result with output of ConvInteger
        # and make the output of this node the same as output of original conv node.
        output_scale_mul_op = ""
        if conv_integer_name != "":
            output_scale_mul_op = conv_integer_name + "_output_scale_mul"
        nodes.append(_get_mul_node([cast_op_output, scales_mul_op_output], node.output[0], output_scale_mul_op))

        return nodes

    def _quantize_matmul_integer_ops(self, node, new_nodes_list):
        '''
        Used when self.mode is QuantizationMode.IntegerOps.
            parameter node: MatMul node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized MatMul node.
        '''
        assert (node.op_type == "MatMul")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], new_nodes_list)

        matmul_integer_output = node.output[0] + "_quantized"
        matmul_integer_name = ""
        if node.name != "":
            matmul_integer_name = node.name + "_quant"
        matmul_integer_node = onnx.helper.make_node("MatMulInteger", quantized_input_names + zero_point_names,
            [matmul_integer_output], matmul_integer_name)
        nodes.append(matmul_integer_node)

        # Add cast operation to cast matmulInteger output to float.
        cast_op_output = matmul_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node("Cast", [matmul_integer_output], [cast_op_output],
            matmul_integer_output + "_cast", to=onnx_proto.TensorProto.FLOAT)
        nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        assert (len(scale_names) == 2)
        if matmul_integer_name != "":
            scales_mul_op = matmul_integer_name + "_scales_mul"
        else:
            scales_mul_op = scale_names[0] + "_" + scale_names[1] + "_mul"

        scales_mul_node = _find_node_by_name(scales_mul_op, self.model.graph, new_nodes_list)
        if scales_mul_node is None:
            scales_mul_node = _get_mul_node(scale_names, scales_mul_op + ":0", scales_mul_op)
            nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        # Add mul operation to multiply mul_scales_op result with output of MatMulInteger
        # and make the output of this node the same as output of original matmul node.
        output_scale_mul_op = ""
        if matmul_integer_name != "":
            output_scale_mul_op = matmul_integer_name + "_output_scale_mul"
        nodes.append(_get_mul_node([cast_op_output, scales_mul_op_output], node.output[0],
            output_scale_mul_op))
        return nodes

    def _quantize_convolution_qlinear_ops(self, node, new_nodes_list):
        '''
        Used when self.mode is QuantizationMode.QLinearOps.
            parameter node: Conv node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "Conv")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], new_nodes_list, TensorLayout.NHWC if self.force_nhwc_conv else TensorLayout.NCHW)

        if self.force_nhwc_conv and self._get_tensor_layout(node.input[0]) != TensorLayout.NHWC:
            (nhwc_input, transpose) = self._convert_to_nhwc(quantized_input_names[0])
            quantized_input_names[0] = nhwc_input
            nodes.extend(transpose)

        # Ensure that the weight input to qlinearconv comes from an initializer
        assert(not self.force_nhwc_conv or \
            (self._get_tensor_layout(node.input[1]) == TensorLayout.NHWC and \
             self.quantized_value_map[node.input[1]].qType == QuantizedValueType.Initializer))
        
        quantized_bias_name = ""
        bias_present = False
        if len(node.input) == 3:
            quantized_bias_name = self._quantize_bias(node, nodes)
            bias_present = True        
        data_found, output_scale_name, output_zp_name, output_scale_shape, output_zp_shape = \
            self._get_quantization_params(node.output[0])

        assert(data_found)

        qlinear_conv_output = node.output[0] + "_quantized"
        qlinear_conv_name = ""
        if node.name != "":
            qlinear_conv_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(_attribute_to_kwarg(attribute))
        qlinear_conv_inputs = []
        # Input 0
        qlinear_conv_inputs.append(quantized_input_names[0])
        qlinear_conv_inputs.append(scale_names[0])
        qlinear_conv_inputs.append(zero_point_names[0])
        # Input 1
        qlinear_conv_inputs.append(quantized_input_names[1])
        qlinear_conv_inputs.append(scale_names[1])
        qlinear_conv_inputs.append(zero_point_names[1])

        # Output
        qlinear_conv_inputs.append(output_scale_name)
        qlinear_conv_inputs.append(output_zp_name)

        if bias_present:
            qlinear_conv_inputs.append(quantized_bias_name)

        qlinear_conv_node = onnx.helper.make_node("QLinearConv_nhwc" if self.force_nhwc_conv else "QLinearConv", qlinear_conv_inputs,
            [qlinear_conv_output], qlinear_conv_name, **kwargs)
        nodes.append(qlinear_conv_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], qlinear_conv_output, output_scale_name, output_zp_name, QuantizedValueType.Input, qType=self.weight_qType, tensor_layout=TensorLayout.NHWC if self.force_nhwc_conv else TensorLayout.NCHW)        
        self.quantized_value_map[node.output[0]] = q_output
        
        return nodes

    def _quantize_matmul_qlinear_ops(self, node, new_nodes_list):
        '''
        Used when self.mode is QuantizationMode.QLinearOps.
            parameter node: MatMul node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Conv node.
        '''
        assert (node.op_type == "MatMul")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self._quantize_inputs(node, [0, 1], new_nodes_list)

        data_found, output_scale_name, output_zp_name, output_scale_shape, output_zp_shape = \
            self._get_quantization_params(node.output[0])
        
        assert(data_found)

        qlinear_matmul_output = node.output[0] + "_quantized"
        qlinear_matmul_name = ""
        if node.name != "":
            qlinear_matmul_name = node.name + "_quant"

        for i in (0, 1):
            if self._get_tensor_layout(node.input[i]) != TensorLayout.NCHW:
                (nchw_input, transpose) = self._convert_to_nchw(quantized_input_names[i])
                quantized_input_names[i] = nchw_input
                nodes.extend(transpose)

        qlinear_matmul_inputs = []
        # Input 0
        qlinear_matmul_inputs.append(quantized_input_names[0])
        qlinear_matmul_inputs.append(scale_names[0])
        qlinear_matmul_inputs.append(zero_point_names[0])
        # Input 1
        qlinear_matmul_inputs.append(quantized_input_names[1])
        qlinear_matmul_inputs.append(scale_names[1])
        qlinear_matmul_inputs.append(zero_point_names[1])
        # Output
        qlinear_matmul_inputs.append(output_scale_name)
        qlinear_matmul_inputs.append(output_zp_name)

        qlinear_matmul_node = onnx.helper.make_node("QLinearMatMul", qlinear_matmul_inputs,
            [qlinear_matmul_output], qlinear_matmul_name)
        nodes.append(qlinear_matmul_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], qlinear_matmul_output, output_scale_name, output_zp_name, QuantizedValueType.Input, qType=self.weight_qType)     
        self.quantized_value_map[node.output[0]] = q_output
        
        return nodes

    def _quantize_convolution(self, node, new_nodes_list):
        '''
            https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
            :param node: Conv node
            :param new_nodes_list: List of new nodes created before processing this node.
            :return: a list of nodes in topological order that represents quantized Conv node
        '''
        assert (node.op_type == "Conv")

        if self.mode == QuantizationMode.IntegerOps:
            return self._quantize_convolution_integer_ops(node, new_nodes_list)

        if self.mode == QuantizationMode.QLinearOps:
            return self._quantize_convolution_qlinear_ops(node, new_nodes_list)

        return [node]

    def _quantize_matmul(self, node, new_nodes_list):
        '''
            https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
            :param node: MatMul node
            :param new_nodes_list: List of new nodes created before processing this node.
            :return: a list of nodes in topological order that represents quantized MatMul node
        '''
        assert(node.op_type == 'MatMul')

        if self.mode == QuantizationMode.IntegerOps:
            return self._quantize_matmul_integer_ops(node, new_nodes_list)

        if self.mode == QuantizationMode.QLinearOps:
            return self._quantize_matmul_qlinear_ops(node, new_nodes_list)

        return [node]

def check_opset_version(org_model, force_fusions):
    '''
        Check opset version of original model and set opset version and fuse_dynamic_quant accordingly.
        If opset version < 10, set quantized model opset version to 10.
        If opset version == 10, do quantization without using dynamicQuantizeLinear operator.
        If opset version == 11, do quantization using dynamicQuantizeLinear operator.

        :return: fuse_dynamic_quant boolean value.
    '''
    global onnx_op_set_version
    opset_version = org_model.opset_import[0].version
    fuse_dynamic_quant = False

    if opset_version < 11 and force_fusions == True:
        print("Warning: The original model opset version is {}, which does not support node fusions.\n\
            Forcing fusions can break other nodes in the model.".format(opset_version))
        onnx_op_set_version = 11
        fuse_dynamic_quant = True
        return fuse_dynamic_quant

    if opset_version < 10:
        print("Warning: The original model opset version is {}, which does not support quantized operators.\n\
            The opset version of quantized model will be set to 10. Use onnx model checker to verify model after quantization.".format(opset_version))
        onnx_op_set_version = 10
    elif opset_version == 10:
        onnx_op_set_version = 10
    else:
        fuse_dynamic_quant = True
    return fuse_dynamic_quant

def quantize(model, per_channel=False, nbits=8, quantization_mode=QuantizationMode.IntegerOps,
    static=False, force_fusions=False, asymmetric_input_types=False, 
    quantization_params=None, nodes_to_quantize=None, force_nhwc_conv=False):
    '''
        Given an onnx model, create a quantized onnx model and save it into a file

    :param model: ModelProto to quantize
    :param per_channel: quantize weights per channel
    :param nbits: number of bits to represent quantized data. Currently only supporting 8-bit types
    :param quantization_mode: Can be one of the QuantizationMode types.
        IntegerOps:
            the function will use integer ops. Only ConvInteger and MatMulInteger ops are supported now.
        QLinearOps:
            the function will use QLinear ops. Only QLinearConv and QLinearMatMul ops are supported now.
    :param static:
        True: The inputs/activations are quantized using static scale and zero point values
              specified through quantization_params.
        False: The inputs/activations are quantized using dynamic scale and zero point values
               computed while running the model.
    :param force_fusions:
        True: Fuses nodes added for dynamic quantization
        False: No fusion is applied for nodes which are added for dynamic quantization.
        Should be only used in cases where backends want to apply special fusion routines
    :param asymmetric_input_types:
        True: Weights are quantized into signed integers and inputs/activations into unsigned integers.
        False: Weights and inputs/activations are quantized into unsigned integers.
    :param quantization_params:
        Dictionary to specify the zero point and scale values for inputs to conv and matmul nodes.
        Should be specified when static is set to True.
        The quantization_params should be specified in the following format:
            {
                "input_name": [zero_point, scale]
            }.
        zero_point should be of type np.uint8 and scale should be of type np.float32.
        example:
            {
                'resnet_model/Relu_1:0': [np.uint8(0), np.float32(0.019539741799235344)],
                'resnet_model/Relu_2:0': [np.uint8(0), np.float32(0.011359662748873234)]
            }    
    :return: ModelProto with quantization
    :param nodes_to quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        exmaple:
        [
            'Cov__224',
            'Conv__252'
        ]
    '''
    if nbits == 8:
        input_qType = onnx_proto.TensorProto.INT8
        weight_qType = onnx_proto.TensorProto.INT8
        mode = quantization_mode
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(model)
        fuse_dynamic_quant = check_opset_version(copy_model, force_fusions)
        quantizer = ONNXQuantizer(copy_model, per_channel, mode, static, fuse_dynamic_quant, weight_qType, input_qType,
                        quantization_params, nodes_to_quantize, force_nhwc_conv)
        quantizer.quantize_model()
        quantizer.model.producer_name = __producer__
        quantizer.model.producer_version = __version__
        return quantizer.model
    else:
        raise ValueError('Unknown value for nbits. only 8 bit quantization is currently supported')