import onnx
import argparse
from pathlib import Path
from onnx import optimizer
from quantizer.quant_utils import generate_identified_filename
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
from onnx import numpy_helper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    parser.add_argument("--replace_gemm", default=False, action='store_true', help="Whether to run gemm to matmul replacement")
    args = parser.parse_args()
    return args

def optimize_model(model_path: Path):
    '''
        Generate model that applies graph optimization (constant folding,etc.)
        parameter model_path: path to the original onnx model
        return: optimized onnx model
    '''
    opt_model_path = generate_identified_filename(model_path, "-opt")
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    _ = InferenceSession(model_path.as_posix(), sess_option)
    optimized_model = onnx.load(opt_model_path.as_posix())
    return optimized_model

def find_by_name(item_name, item_list):
    '''
    Helper function to find item by name in a list.
        parameter item_name: name of the item.
        parameter item_list: list of items.
        return: item if found. None otherwise.
    '''
    items = [item for item in item_list if item.name == item_name]
    return items[0] if len(items) > 0 else None

def add_initializer(model, tensor):
    if find_by_name(tensor.name, model.graph.initializer) is None:
        model.graph.initializer.extend([tensor])

def get_initializer(model, name):
    for tensor in model.graph.initializer:
        if tensor.name == name:
            return tensor
    return None

def remove_initializer(model, tensor):
    if tensor in model.graph.initializer:
        model.graph.initializer.remove(tensor)
    for idx, init in enumerate(model.graph.input):
        if init.name == tensor.name:
            del model.graph.input[idx]
            return

def replace_gemm_with_matmul(model):
    new_nodes = []

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
            if alpha == 1.0 and beta == 1.0 and transA == 0:
                print("Replacing gemm with input {}, {}  with matmul.".format(node.input[0], node.input[1]))
                print("Be careful with this since matmuls at the end are very sensitive to quantization")
                inputB = node.input[1]
                if transB == 1:
                    B = get_initializer(model, node.input[1])
                    if B:
                        # assume B is not used by any other node
                        B_array = onnx.numpy_helper.to_array(B)
                        B_trans = onnx.numpy_helper.from_array(B_array.T)
                        B_trans.name = B.name
                        remove_initializer(model, B)
                        add_initializer(model, B_trans)
                    else:
                        inputB += '_Transposed'
                        transpose_node = onnx.helper.make_node('Transpose',
                                                            inputs=[node.input[1]],
                                                            outputs=[inputB],
                                                            name=node.name+'_Transpose')
                        new_nodes.append(transpose_node)

                matmul_node = onnx.helper.make_node('MatMul',
                                                    inputs=[node.input[0], inputB],
                                                    outputs=[node.output[0] + ('_MatMul' if len(node.input)>2 else '')],
                                                    name=node.name + '_MatMul')
                new_nodes.append(matmul_node)

                if len(node.input) > 2:
                    add_node = onnx.helper.make_node('Add',
                                                        inputs=[node.output[0] + '_MatMul', node.input[2]],
                                                        outputs=node.output,
                                                        name=node.name + '_Add')
                    new_nodes.append(add_node)  
            
            # unsupported
            else:
                new_nodes.append(node)
        
        # not GEMM
        else:
            new_nodes.append(node)

    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)

def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Warning: Model with ir_version below 4 requires initializer in graph input, so not removing'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])


def add_value_info_for_constants(model):
    """
    https://github.com/onnx/onnx/issues/2995
    
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    add_const_value_infos_to_graph(model.graph)


def sum_to_add(model):
    '''
    Convert an add with 2 operators into a sum
    '''
    for node in model.graph.node:
        if node.op_type == 'Sum' and len(node.input) == 2:
            node.op_type = 'Add'

def transforms():
    args = get_args()
    model = onnx.load(args.input)

    # https://github.com/onnx/onnx/issues/2902
    add_value_info_for_constants(model)
    for init in model.graph.initializer:
        for value_info in model.graph.value_info:
            if init.name == value_info.name:
                model.graph.input.append(value_info)

    optimized = optimizer.optimize(model, ['extract_constant_to_initializer', 'fuse_bn_into_conv'])
    remove_initializer_from_input(optimized)
    if args.replace_gemm:
        replace_gemm_with_matmul(optimized)
    sum_to_add(optimized)
    onnx.save(optimized, args.output)
    print("Done")

if __name__ == '__main__':
    transforms()