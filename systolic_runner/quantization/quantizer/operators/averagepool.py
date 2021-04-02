import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType
from onnx import onnx_pb as onnx_proto
import numpy as np

def tensor_shape_to_list(tensor_type):
    """ Convert tensor shape to list
    """
    shape_list = []
    for d in tensor_type.shape.dim:
        if (d.HasField("dim_value")):
            shape_list.append(d.dim_value)  # known dimension
        elif (d.HasField("dim_param")):
            shape_list.append(d.dim_param)  # unknown dimension with symbolic name
        else:
            shape_list.append("?")  # shall not happen
    return shape_list

class QAveragePool(QuantOperatorBase):
    '''
    We convert an averagepool op into a strided dw-convolution if it is supported
    '''
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "AveragePool")

        can_convert_to_conv = True
        pads = []
        strides = []
        kernel_shape = []

        for attr in node.attribute:
            if attr.name == 'auto_pad':
                val = onnx.helper.get_attribute_value(attr)
                if val != "NOTSET":
                    can_convert_to_conv = False
                    break
            elif attr.name in ['ceil_mode', 'count_include_pad']:
                val = onnx.helper.get_attribute_value(attr)
                if val != 0:
                    can_convert_to_conv = False
                    break
            elif attr.name == 'pads':
                pads = onnx.helper.get_attribute_value(attr)
            elif attr.name == 'strides':
                strides = onnx.helper.get_attribute_value(attr)
            elif attr.name == 'kernel_shape':
                kernel_shape = onnx.helper.get_attribute_value(attr)

        input_shape = []
        if node.input[0] not in self.quantizer.value_infos:
            can_convert_to_conv = False
        else:
            input_shape = tensor_shape_to_list(self.quantizer.value_infos[node.input[0]].type.tensor_type)
            can_convert_to_conv = can_convert_to_conv and len(input_shape) == 4
        
        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0])

        # If we cannot quantize it, skip
        if node.input[0] not in self.quantizer.quantized_value_map or \
            not can_convert_to_conv or \
            not data_found or \
            self.quantizer.quantization_params[node.output[0]][0].item() != 0:
            return super().quantize()

        groups = input_shape[1]
        numpy_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[self.quantizer.input_qType]
        weight_matrix = np.ones((groups, 1, kernel_shape[0], kernel_shape[1]), dtype=numpy_type)
        w_scale = np.array([1], dtype=np.float32)
        w_zero_point = np.array([0], dtype=numpy_type)

        packed_weight_initializer = onnx.numpy_helper.from_array(weight_matrix, node.input[0] + "_1s_weight_vals")
        weight_scale_initializer = onnx.helper.make_tensor(node.input[0] + "_1s_weight_scale",
                                                                     onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[w_scale.dtype], [], w_scale.tolist())
        weight_zp_initializer = onnx.helper.make_tensor(node.input[0] + "_1s_weight_zp",
                                                                     onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[w_zero_point.dtype], [], w_zero_point.tolist())
        
        
        adjusted_output_scale_initializer = onnx.helper.make_tensor(output_scale_name + "_avg_pool", onnx_proto.TensorProto.FLOAT, [], 
                                             [self.quantizer.quantization_params[node.output[0]][1].item() * kernel_shape[0] * kernel_shape[1]])
        

        self.quantizer.model.initializer().extend([packed_weight_initializer, weight_scale_initializer, weight_zp_initializer, adjusted_output_scale_initializer])

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [0])

        qlinearconv_op_inputs = []

        qlinearconv_op_inputs.append(quantized_input_names[0])
        qlinearconv_op_inputs.append(scale_names[0])
        qlinearconv_op_inputs.append(zero_point_names[0])

        qlinearconv_op_inputs.append(node.input[0] + "_1s_weight_vals")
        qlinearconv_op_inputs.append(node.input[0] + "_1s_weight_scale")
        qlinearconv_op_inputs.append(node.input[0] + "_1s_weight_zp")

        qlinearconv_op_inputs.append(output_scale_name + "_avg_pool")
        qlinearconv_op_inputs.append(output_zp_name)

        qlinearconv_output = node.output[0] + "_quantized"
        qlinearconv_name = node.name + "_quant" if node.name != "" else ""

        
        kwargs = {}
        
        for (k, v) in {"pads": pads, "strides": strides, "group": groups}.items():
            if v:
                kwargs[k] = v

        qlinearconv_op = onnx.helper.make_node("QLinearConv", qlinearconv_op_inputs,
                                                         [qlinearconv_output], qlinearconv_name,
                                                         **kwargs)
        nodes.append(qlinearconv_op)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], qlinearconv_output, output_scale_name, output_zp_name,
                                  QuantizedValueType.Input, qType=self.quantizer.input_qType)
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes
