import onnx
from .base_operator import QuantOperatorBase
from onnx import onnx_pb as onnx_proto
from onnx import helper
import numpy as np
import math
'''
Quantize Scan
'''

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

def parse_scale_from_line(line):
    line = line.split(" ")
    op_name = line[0]
    qkv_type = int(line[1])
    mn = float(line[2])
    mx = float(line[3])
    return op_name, mn, mx

def get_scales():
    FILE = "scan_op.txt"
    scales = {}
    with open(FILE) as scalefile:
        all_lines = scalefile.readlines()
        for op_name, mn, mx in map(parse_scale_from_line, all_lines):
            if op_name in scales:
                curmin, curmax = scales[op_name]
                scales[op_name] = (min(curmin, mn), max(curmax, mx))
            else:
                scales[op_name] = (mn, mx)

    return {k : calculate_scale_zeropoint(scales[k][0], scales[k][1], 'int8') for k in scales.keys()}

scales = get_scales()


class Scan(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def get_attribute(node, attr_name, default_value=None):
        found = [attr for attr in node.attribute if attr.name == attr_name]
        if found:
            return helper.get_attribute_value(found[0])
        return default_value

    def set_graph(node, attr_name, set_value):
        found = [attr for attr in node.attribute if attr.name == attr_name]
        if found:
            found[0].g.CopyFrom(set_value)
            return
        assert False, "Should not reach here"

    def quantize(self):
        node = self.node
        onnx_quantizer = self.quantizer
        assert (node.op_type == "Scan")
        body = Scan.get_attribute(node, "body")

        dummy_model = helper.make_model(body, producer_name='graph_test')
        copy_model = onnx_proto.ModelProto()
        copy_model.CopyFrom(dummy_model)

        # Why .__class__? Because python makes importing stuff harder than it needs to be. 
        scan_quantizer = onnx_quantizer.__class__(copy_model, onnx_quantizer.per_channel, onnx_quantizer.reduce_range,
                                       onnx_quantizer.mode,
                                       False, # = static. Using dynamic quantization for Scan body
                                       onnx_quantizer.weight_qType, onnx_quantizer.input_qType,
                                       scales,
                                       onnx_quantizer.nodes_to_quantize,
                                       onnx_quantizer.nodes_to_exclude,
                                       onnx_quantizer.op_types_to_quantize)

        scan_quantizer.quantize_model()

        copy_node = onnx_proto.NodeProto()
        copy_node.CopyFrom(node)
        Scan.set_graph(copy_node, "body", scan_quantizer.model.model.graph)

        self.quantizer.new_nodes += [copy_node]
