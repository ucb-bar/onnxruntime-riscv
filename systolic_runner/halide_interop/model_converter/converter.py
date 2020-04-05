from model import Model
from onnx import helper
from onnx import TensorProto, shape_inference
import onnx.utils
import numpy as np
import argparse
import os
from fnv import hash_dn
import cpp_templates
from cpp_templates import header, kernel_def, op_def, register_single, register, custom_format
from onnx_types import VI

NODES_TO_HALIDE = ['LRN', 'MaxPool', 'Relu', 'Add']
GENERATED_PREFIX = "./generated/"

value_info_dict = {}


kernel_specs = []
op_specs = []
node_names = []

def build_kernelspec(node_name, inputs, outputs):
    include = '#include "{}.h"\n'.format(node_name)
    prep_inputs = []
    for idx, inp in enumerate(inputs):
        vi = VI(value_info_dict[inp].type)
        prep_inputs += ["const OrtValue* input_{id} = ort_.KernelContext_GetInput(context, {id});".format(id=str(idx))]
        prep_inputs += ["Halide::Runtime::Buffer<const {type}> input_{id}_buf = getInBufferForOrtValue<{type}>(ort_, input_{id});".format(type=vi.t.c, id=idx)]

    prep_outputs = []
    for idx, out in enumerate(outputs):
        vi = VI(value_info_dict[out].type)
        prep_outputs += ["int64_t output_{id}_dims[] = {lst};".format(id=idx, lst=str(vi.shape).replace('[', '{').replace(']', '}'))]
        prep_outputs += ["OrtValue* output_{id} = ort_.KernelContext_GetOutput(context, {id}, output_{id}_dims, {dims});".format(id=idx, dims=vi.dims)]
        prep_outputs += ["Halide::Runtime::Buffer<{type}> output_{id}_buf = getOutBufferForOrtValue<{type}>(ort_, output_{id}, {lst});".format(type=vi.t.c, id=idx, lst=str(vi.shape).replace('[', '{').replace(']', '}'))]

    func_call = "{name}({args});".format(name=node_name, args=', '.join(["input_" + str(i) + "_buf" for i in range(len(inputs))] + ["output_" + str(i) + "_buf" for i in range(len(outputs))]))
    
    kern = custom_format(kernel_def, "@$", kernel_name=node_name, setup_input="\n\t".join(prep_inputs), setup_output="\n\t".join(prep_outputs), call_function=func_call)
    return include + kern

def build_opspec(node_name, inputs, outputs):
    switch_in = ["switch(index) {"]
    for idx, inp in enumerate(inputs):
        switch_in += ["case {}: return ONNX_TENSOR_ELEMENT_DATA_TYPE_{};".format(idx,  VI(value_info_dict[inp].type).t.onnx_str)]
    switch_in += ["}"]
    switch_out = ["switch(index) {"]
    for idx, out in enumerate(outputs):
        switch_out += ["case {}: return ONNX_TENSOR_ELEMENT_DATA_TYPE_{};".format(idx, VI(value_info_dict[out].type).t.onnx_str)]
    switch_out += ["}"]
    return custom_format(op_def, "@$", kernel_name=node_name, num_inputs=str(len(inputs)), num_outputs=str(len(outputs)), input_type="\n\t".join(switch_in), output_type="\n\t".join(switch_out))

def gen_file(node):
    global generated_file, kernel_specs, op_specs, node_names

    for inp in node.input:
        if (inp not in value_info_dict):
            raise ValueError("Could not infer shape for {}".format(inp))
    for output in node.output:
        if output not in value_info_dict:
            raise ValueError("Could not infer shape for {}".format(output))

    node_name = node.op_type + "_"+ hash_dn(node.op_type + ''.join(node.input) + ''.join(node.output))

    graph_def = helper.make_graph([node], node_name, [value_info_dict[inp] for inp in node.input],
                                  [value_info_dict[output] for output in node.output])
    onnx_model = helper.make_model(graph_def, producer_name='onnx-example')
    model = Model()
    model.BuildFromOnnxModel(onnx_model)
    model.OptimizeSchedule()
    
    model.Compile(func_name=node_name, lib_name=GENERATED_PREFIX + node_name)

    node.op_type = "CustomOp" + node_name
    node.domain = "test.customop"

    kernel_specs += [build_kernelspec(node_name, node.input, node.output)]
    op_specs += [build_opspec(node_name, node.input, node.output)]
    node_names += [node_name]

def add_value_info_for_constants(model : onnx.ModelProto):
    """
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


def main(model=None):
    global value_info_dict

    try:
        if not os.path.exists(GENERATED_PREFIX):
            os.makedirs(GENERATED_PREFIX)
    except OSError:
        print ('Error: Creating directory. ' +  GENERATED_PREFIX)
        
    if model is None:
        parser = argparse.ArgumentParser(description='input model path')
        parser.add_argument('--model_path', required=True)
        args = parser.parse_args()

        model = onnx.load(args.model_path)
        if model.ir_version <= 3:
            model.ir_version = 4
        add_value_info_for_constants(model)
        if len(model.opset_import) == 1:
            model.opset_import.append(helper.make_operatorsetid("", 10))

    polished_model = shape_inference.infer_shapes(model)
    name_to_value_info = {x.name: x for x in polished_model.graph.value_info}
    input_value_info = {x.name : x for x in polished_model.graph.input}
    output_value_info = {x.name : x for x in polished_model.graph.output}

    value_info_dict = {**name_to_value_info, **input_value_info, **output_value_info}
    for idx, node in enumerate(polished_model.graph.node):
        if node.op_type in NODES_TO_HALIDE:
            gen_file(node)

    with open(GENERATED_PREFIX + "custom_op_library.cc", "w+") as f:
        f.write(header)
        f.write("\n".join(kernel_specs))
        f.write("\n".join(op_specs))
        registers = []
        for node in node_names:
            registers += [custom_format(register_single, "@$", kernel_name=node)]
        f.write(custom_format(register, "@$", register_kernels="\n".join(registers)))

    onnx.save(polished_model, GENERATED_PREFIX + "model_rewritten.onnx")


if __name__ == '__main__':
    main()

    # A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 64, 112, 112])
    # B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 64, 112, 112])
    # # Create one output
    # C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 64, 112, 112])

    # D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 64, 112, 112])
    # # Create a node
    # node_def = helper.make_node('Add', ['A', 'B'], ['C'])
    # node_def2 = helper.make_node('Relu', ['C'], ['D'])

    # # Create the model
    # graph_def = helper.make_graph([node_def], "scalar-model", [A, B], [C])
    # onnx_model = helper.make_model(graph_def, producer_name='onnx-example')
    # main(onnx_model)

