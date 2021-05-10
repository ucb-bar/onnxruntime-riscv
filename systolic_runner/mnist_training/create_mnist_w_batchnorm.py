import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import onnx.optimizer

import math
import numpy as np
import scipy.stats as stats

def truncated_normal(dims):   
    size = 1
    for dim in dims:
        size *= dim

    mu, stddev = 0, 1/math.sqrt(size)
    lower, upper = -2 * stddev, 2 * stddev
    X = stats.truncnorm( (lower - mu) / stddev, (upper - mu) / stddev, loc = mu, scale = stddev)

    return X.rvs(size).tolist()

    
def zeros(dim):
    return [0] * dim[0]

batch_size = -1

W1_dims = [8, 1, 5, 5]
W2_dims = [16, 8, 5, 5]
W3_dims = [256, 10]

W1 =  helper.make_tensor(name="W1", data_type=onnx.TensorProto.FLOAT, dims=W1_dims, vals=truncated_normal(W1_dims))
W2 =  helper.make_tensor(name="W2", data_type=onnx.TensorProto.FLOAT, dims=W2_dims, vals=truncated_normal(W2_dims))
W3 =  helper.make_tensor(name="W3", data_type=onnx.TensorProto.FLOAT, dims=W3_dims, vals=truncated_normal(W3_dims))

B1_dims = [8]
B2_dims = [16]
B3_dims = [10]

B1 =  helper.make_tensor(name="B1", data_type=onnx.TensorProto.FLOAT, dims=B1_dims, vals=zeros(B1_dims))
B2 =  helper.make_tensor(name="B2", data_type=onnx.TensorProto.FLOAT, dims=B2_dims, vals=zeros(B2_dims))
B3 =  helper.make_tensor(name="B3", data_type=onnx.TensorProto.FLOAT, dims=B3_dims, vals=zeros(B3_dims))
foo = lambda x: [0.5]*8
s = helper.make_tensor(name='s', data_type=onnx.TensorProto.FLOAT, dims=[8], vals=foo([8]))

bias = helper.make_tensor(name='bias', data_type=onnx.TensorProto.FLOAT, dims=[8], vals=foo([8]))
mean = helper.make_tensor(name='mean', data_type=onnx.TensorProto.FLOAT, dims=[8], vals=foo([8]))
var = helper.make_tensor(name='var', data_type=onnx.TensorProto.FLOAT, dims=[8], vals=foo([8]))

shape = helper.make_tensor(name="shape", data_type=onnx.TensorProto.INT64, dims=[2], vals=[batch_size, 256])

node0 = helper.make_node('BatchNormalization', inputs=['T1', 's', 'bias', 'mean', 'var'], outputs=['T1_bn'])

node1 = helper.make_node('Conv', inputs=['X', 'W1', 'B1'], outputs=['T1'], kernel_shape=[5,5], strides=[1,1], pads=[2,2,2,2])
node2 = helper.make_node('Relu', inputs=['T1_bn'], outputs=['T2'])
node3 = helper.make_node('MaxPool', inputs=['T2'], outputs=['T3'], kernel_shape=[2,2], strides=[2,2])

node4 = helper.make_node('Conv', inputs=['T3', 'W2', 'B2'], outputs=['T4'], kernel_shape=[5,5], strides=[1,1], pads=[2,2,2,2])
node5 = helper.make_node('Relu', inputs=['T4'], outputs=['T5'])
node6 = helper.make_node('MaxPool', inputs=['T5'], outputs=['T6'], kernel_shape=[3,3], strides=[3,3])

node7 = helper.make_node('Reshape', inputs=['T6', 'shape'], outputs=['T7'])

node8 = helper.make_node('Gemm', inputs=['T7', 'W3', 'B3'], outputs=['predictions'])

graph = helper.make_graph(
    [node1, node0, node2, node3, node4, node5, node6, node7, node8],
    'mnist_conv',
    [ helper.make_tensor_value_info('s', TensorProto.FLOAT, ([8])),
     helper.make_tensor_value_info('bias', TensorProto.FLOAT, ([8])),
     helper.make_tensor_value_info('mean', TensorProto.FLOAT, ([8])),
    helper.make_tensor_value_info('var', TensorProto.FLOAT, ([8])),
     helper.make_tensor_value_info('X', TensorProto.FLOAT, ([batch_size, 1, 28, 28])),
     helper.make_tensor_value_info('W1', TensorProto.FLOAT, W1_dims),
     helper.make_tensor_value_info('W2', TensorProto.FLOAT, W2_dims),
     helper.make_tensor_value_info('W3', TensorProto.FLOAT, W3_dims),
     helper.make_tensor_value_info('B1', TensorProto.FLOAT, B1_dims),
     helper.make_tensor_value_info('B2', TensorProto.FLOAT, B2_dims),
     helper.make_tensor_value_info('B3', TensorProto.FLOAT, B3_dims),
     helper.make_tensor_value_info('shape', TensorProto.INT64, [2]),
    ],
    [helper.make_tensor_value_info('predictions', TensorProto.FLOAT, ([batch_size, 10]))],
    [s, bias, mean, var, W1, W2, W3, B1, B2, B3, shape]
)
original_model = helper.make_model(graph, producer_name='onnx-examples')

onnx.checker.check_model(original_model)
inferred_model = shape_inference.infer_shapes(original_model)
onnx.save_model(inferred_model, "mnist_conv_w_batchnorm.onnx")
