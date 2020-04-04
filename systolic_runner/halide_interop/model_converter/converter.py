from model import Model
from onnx import helper
from onnx import TensorProto
import onnx.utils
import numpy as np
import argparse
import os
from fnv import hash_dn
import cpp_templates
from cpp_templates import header, kernel_def, op_def, register_single, register, custom_format
from onnx_types import VI

NODES_TO_HALIDE = ['LRN', 'MaxPool']
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

    node_name = hash_dn(node.op_type + ''.join(node.input) + ''.join(node.output))

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



def main():
    global value_info_dict
    parser = argparse.ArgumentParser(description='input model path')
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()

    try:
        if not os.path.exists(GENERATED_PREFIX):
            os.makedirs(GENERATED_PREFIX)
    except OSError:
        print ('Error: Creating directory. ' +  GENERATED_PREFIX)

    model = onnx.load(args.model_path)
    polished_model = onnx.utils.polish_model(model)
    name_to_value_info = {x.name: x for x in polished_model.graph.value_info}
    input_value_info = {x.name : x for x in polished_model.graph.input}
    output_value_info = {x.name : x for x in polished_model.graph.output}

    value_info_dict = {**name_to_value_info, **input_value_info, **output_value_info}
    gen_file(polished_model.graph.node[0])
    gen_file(polished_model.graph.node[1])

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