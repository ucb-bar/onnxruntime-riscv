from calibrate import calculate_scale_zeropoint
import onnx
import argparse
from onnx import onnx_pb as onnx_proto
from quantizer.onnx_model import ONNXModel
import pprint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input file")
    parser.add_argument("--output", required=True, help="output dict")
    args = parser.parse_args()
    return args

def parse_scale_from_line(line):
    line = line.split(" ")
    op_name = line[0]
    qkv_type = int(line[1])
    mn = float(line[2])
    mx = float(line[3])
    return op_name, mn, mx

def main():
    args = get_args()
    scales = {}
    with open(args.input) as scalefile:
        all_lines = scalefile.readlines()
        for op_name, mn, mx in map(parse_scale_from_line, all_lines):
            if op_name in scales:
                curmin, curmax = scales[op_name]
                scales[op_name] = (min(curmin, mn), max(curmax, mx))
            else:
                scales[op_name] = (mn, mx)

    pprint.pprint({k : calculate_scale_zeropoint([], scales[k][0], scales[k][1], 'int8') for k in scales.keys()})

    

if __name__ == '__main__':
    main()