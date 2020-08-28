import onnx
import argparse
from onnx import optimizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    args = parser.parse_args()
    return args

def replace_gemm_with_matmul(model):
    nodes_to_remove = []
    nodes_to_add = []
    for node in model.graph.node:
        if node.op_type == 'Gemm':
            alpha = 1.0
            beta = 1.0
            transA = 0
            transB = 0
            for attr in node.attribute:
                if attr.name == 'alpha':
                    alpha = onnx.helper.get_attribute_value(attr)
                elif attr.name == 'beta':
                    beta = onnx.helper.get_attribute_value(attr)
                elif attr.name == 'transA':
                    transA = onnx.helper.get_attribute_value(attr)
                elif attr.name == 'transB':
                    transB = onnx.helper.get_attribute_value(attr)
            if alpha == 1.0 and beta == 1.0 and transA == 0 and transB == 0:
                matmul_node = onnx.helper.make_node(
                    'MatMul',
                    [node.input[0], node.input[1]],
                    [node.output[0]+'_MatMul'],
                    name=node.output[0]+'_MatMul')

                add_node = onnx.helper.make_node(
                    'Add',
                    inputs=[node.output[0]+'_MatMul', node.input[2]],
                    outputs=node.output,
                    name=node.output[0]+'_Add')

                nodes_to_remove.extend([node])
                nodes_to_add.extend([matmul_node, add_node])

    model.graph.node.extend(nodes_to_add)
    for node in nodes_to_remove:
        model.graph.node.remove(node)

def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])


def transforms():
    args = get_args()
    model = onnx.load(args.input)
    optimized = optimizer.optimize(model, ['extract_constant_to_initializer'])
    remove_initializer_from_input(optimized)
    replace_gemm_with_matmul(optimized)
    onnx.save(optimized, args.output)

if __name__ == '__main__':
    transforms()