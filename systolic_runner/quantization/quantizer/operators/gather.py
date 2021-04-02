import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType
from onnx import onnx_pb as onnx_proto
'''
    Quantize Gather
'''


class GatherQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)
        self.input_name_to_nodes = self.quantizer.model.input_name_to_nodes()

    def quantize(self):
        node = self.node
        assert (node.op_type == "Gather")

        # Don't bother quantizing if input not already quantized as might reduce performance
        if node.input[0] not in self.quantizer.quantized_value_map:
            return super().quantize()

        # For now we only handle gather if the next node are all shape. Because in that case the scale doesn't matter
        # Theoretically, we should be safe even if this isn't the case because this operation
        # Just selects a subset of input indices, so we should be safe letting the scale of the output be the same
        # as the scale of the input. I.e. if the inputs are within the rage of the input scale factor
        # It is guaranteed that the outputs will fit inside the input scale factor as well.
        # (Of course we can potentially do better than using the input scale factor.)
        if not all(x.op_type == "Shape" for x in self.quantizer.model.get_children(node, self.input_name_to_nodes)):
            super().quantize()
            return

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