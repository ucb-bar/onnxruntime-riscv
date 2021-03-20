import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType
from onnx import onnx_pb as onnx_proto


class QNoop(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        # If input to this node is not quantized then keep this node
        if node.input[0] not in self.quantizer.quantized_value_map:
            self.quantizer.new_nodes += [node]
            return

        # Reshape is a no-op in terms of quantization -- preserves scale
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                quantized_input_value.scale_name, quantized_input_value.zp_name,
                                                QuantizedValueType.Input,
                                                qType=quantized_input_value.qType)
        # Create an entry for output quantized value
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        node.input[0] = quantized_input_value.q_name
        node.output[0] = quantized_output_value.q_name
        self.quantizer.new_nodes += [node]


class QShape(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        # If input to this node is not quantized then keep this node
        if node.input[0] not in self.quantizer.quantized_value_map:
            self.quantizer.new_nodes += [node]
            return

        # Shape can accept quant input
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        node.input[0] = quantized_input_value.q_name

        # Output is int64 and does not have to change
        self.quantizer.new_nodes += [node]
    

class QConcat(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        # If all input to this node is not quantized then keep this node
        if not all(inp in self.quantizer.quantized_value_map for inp in node.input):
            self.quantizer.new_nodes += [node]
            return

        for idx, inp in enumerate(node.input):
            quantized_input_value = self.quantizer.quantized_value_map[inp]
            node.input[idx] = quantized_input_value.q_name

        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0])

        if not data_found:
            raise ValueError("Quantization parameters for output:\"{}\" of node:\"{}\" not specified".format(
                node.output[0], node.name))

        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                output_scale_name, output_zp_name,
                                                QuantizedValueType.Input,
                                                qType=quantized_input_value.qType)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value
        node.output[0] = quantized_output_value.q_name

        # Output is int64 and does not have to change
        self.quantizer.new_nodes += [node]