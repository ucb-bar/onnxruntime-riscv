import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, ms_domain, QuantizedValue, QuantizedValueType
from onnx import onnx_pb as onnx_proto


class QLinearBinaryOp(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        # Don't bother quantizing if none of the inputs are quantized as might reduce performance
        if node.input[0] not in self.quantizer.quantized_value_map and \
           node.input[1] not in self.quantizer.quantized_value_map:
            return super().quantize()

        # Check whether all inputs are floats, so we can quantize it
        can_quantize = self.quantizer._is_valid_quantize_value(node.output[0]) and \
                       self.quantizer._is_valid_quantize_value(node.input[0]) and \
                       self.quantizer._is_valid_quantize_value(node.input[1])

        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0])
        if (not data_found or not can_quantize):  # only try to quantize when given quantization parameters for it
            if can_quantize:
                print("Skipping potentially quantizable {} op (name: {}) since output parameter not found".format(node.op_type, node.name))
            return super().quantize()

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [0, 1], initializer_use_weight_qType=False)

        qlinear_binary_math_output = node.output[0] + "_quantized"
        qlinear_binary_math_name = node.name + "_quant" if node.name != "" else ""

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlinear_binary_math_inputs = []
        # Input 0
        qlinear_binary_math_inputs.append(quantized_input_names[0])
        qlinear_binary_math_inputs.append(scale_names[0])
        qlinear_binary_math_inputs.append(zero_point_names[0])
        # Input 1
        qlinear_binary_math_inputs.append(quantized_input_names[1])
        qlinear_binary_math_inputs.append(scale_names[1])
        qlinear_binary_math_inputs.append(zero_point_names[1])

        # Output
        qlinear_binary_math_inputs.append(output_scale_name)
        qlinear_binary_math_inputs.append(output_zp_name)

        qlinear_binary_math_node = onnx.helper.make_node("QLinear" + node.op_type, qlinear_binary_math_inputs,
                                                         [qlinear_binary_math_output], qlinear_binary_math_name,
                                                         **kwargs)
        nodes.append(qlinear_binary_math_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(node.output[0], qlinear_binary_math_output, output_scale_name, output_zp_name,
                                  QuantizedValueType.Input, qType=self.quantizer.input_qType)
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes
