import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType
from onnx import onnx_pb as onnx_proto
import numpy as np
import math


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
    

# The quantization for concat is a bit subtle
# The inputs could potentially all have different scale factors
# So I'm not sure if we can just naively concat the inputs
# Because e.g. consider a situation where you have two inputs
# that are close to saturated, both 127 and 127. But the former has 
# scale factor 0.01, and the later has scale factor 1. Then the true (fp32) output
# is supposed to be {1.27, 1} which again has scale factor 100. 
# But now the quantized result is {127, 100} which isn't just a naive concat
# Moreover, if we naively concat'd but still tried to use the 
# calculatd output scale of 100 found during calibration,
#  we would end up with an equivalent fp32 output of {1.27, 1.27}
# However, we could do something like concat where we quantize concat
# If all inputs have the same scale
class QConcat(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        raise NotImplementedError("This isn't implemented, and likely cannot be. See the comment above")
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

# Quantizing this is again a bit subtle since we run into issues similar to
# Those mentioned for concat. Namely, with scatter if `data` and `update`
# have different scale factors then trying to combine them is nonsensical.
# However,for our specific case of mask-rcnn, both `data` and `update`
# are of the same scale so it suffices
class QScatter(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)
        self.output_name_to_node = self.quantizer.model.output_name_to_node()
        self.input_name_to_nodes = self.quantizer.model.input_name_to_nodes()

    def quantize(self):
        node = self.node

        # We can only quantize if the input scales and output scales are the same
        # Or the input scale for one is a constant 0
        in_params = self.quantizer.quantization_params[node.input[0]]
        out_params = self.quantizer.quantization_params[node.output[0]]
        same_inp_out = in_params[0] == out_params[0] and math.isclose(in_params[1], out_params[1], abs_tol=1e-4)
        parent_node = self.quantizer.model.get_parent(node, 0, self.output_name_to_node)

        constant_in = in_params == [0, 1.0] and \
                      parent_node.op_type  == "ConstantOfShape" and \
                      np.array_equal(onnx.numpy_helper.to_array(parent_node.attribute[0].t), [0])
        if not (same_inp_out or constant_in):
            print("Not quantizing scatter since input/output scale doesn't match")
            super().quantize()
            return

        # Set the scale of the updates to be same as output scale
        self.quantizer.quantization_params[node.input[0]] = self.quantizer.quantization_params[node.output[0]]
        self.quantizer.quantization_params[node.input[2]] = self.quantizer.quantization_params[node.output[0]]

        # If the output of constant 0 only goes to this node
        if constant_in and len(self.quantizer.model.get_children(parent_node, self.input_name_to_nodes)) == 1:
            # Change that to be a tensor of int8
            if node.input[0] in self.quantizer.value_infos:
                for idx, elem in enumerate(self.quantizer.model.model.graph.value_info):
                    if elem.name == node.input[0]:
                        elem.type.tensor_type.elem_type = 3 if in_params[0].dtype == 'int8' else 2
                    

            parent_node.attribute[0].t.CopyFrom(onnx.numpy_helper.from_array(np.array([0], dtype=in_params[0].dtype)))
            (quantized_input_names, zero_point_names, scale_names, nodes) = \
                self.quantizer.quantize_inputs(node, [2])
            quantized_input_names.insert(0, node.input[0])
        else:
            (quantized_input_names, zero_point_names, scale_names, nodes) = \
                self.quantizer.quantize_inputs(node, [0, 2])

        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0])

        if not data_found:
            raise ValueError("Quantization parameters for output:\"{}\" of node:\"{}\" not specified".format(
                node.output[0], node.name))

        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                output_scale_name, output_zp_name,
                                                QuantizedValueType.Input,
                                                qType=self.quantizer.input_qType)
        # Create an entry for output quantized value
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        node.input[0] = quantized_input_names[0]
        node.input[2] = quantized_input_names[1]
        node.output[0] = quantized_output_value.q_name
        self.quantizer.new_nodes += nodes
        self.quantizer.new_nodes += [node]