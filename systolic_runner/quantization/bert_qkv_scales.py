from calibrate import calculate_scale_zeropoint
import onnx
import argparse
from onnx import onnx_pb as onnx_proto
from quantizer.onnx_model import ONNXModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--scales", required=True, help="qkv txt")
    parser.add_argument("--output", required=True, help="output model")
    args = parser.parse_args()
    return args

def parse_scale_from_line(line):
    line = line.split(" ")
    op_name = line[1]
    qkv_type = int(line[2])
    mn = float(line[3])
    mx = float(line[4])
    scale = calculate_scale_zeropoint([], mn, mx, 'int8')[1]
    return op_name + '_quant', {qkv_type: scale}


def update_model(model, scales):
    for node in model.nodes():
        if node.name in scales:
            assert(len(node.input) == 9)
            scale_shape = []
            scale_values = [scales[node.name][0]]
            scale_name = node.name + "_q_scale"
            init_scale = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, scale_values)
            model.add_initializer(init_scale)
            node.input.extend([scale_name])

            scale_values = [scales[node.name][1]]
            scale_name = node.name + "_k_scale"
            init_scale = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, scale_values)
            model.add_initializer(init_scale)
            node.input.extend([scale_name])


            scale_values = [scales[node.name][2]]
            scale_name = node.name + "_v_scale"
            init_scale = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, scale_shape, scale_values)
            model.add_initializer(init_scale)
            node.input.extend([scale_name])


def main():
    args = get_args()
    model = ONNXModel(onnx.load(args.input))
    scales = {}
    with open(args.scales) as scalefile:
        all_lines = scalefile.readlines()
        for op_name, val in map(parse_scale_from_line, all_lines):
            if op_name in scales:
                scales[op_name].update(val)
            else:
                scales[op_name] = val
    update_model(model, scales)
    onnx.save(model.model, args.output)

    

if __name__ == '__main__':
    main()