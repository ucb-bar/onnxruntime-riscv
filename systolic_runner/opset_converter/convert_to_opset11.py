import onnx.numpy_helper
import onnx
import argparse
from onnx import onnx_pb as onnx_proto
from onnx import version_converter

'''
Are you kidding me?
ONNX operators change their specs from version to version. 
Why not just use an op-level version namespace or something?
But no, sadly we have to resort to writing hacky converters to run models.

Slice v10 moves the {axes, ends, starts} attributes of v1 into the inputs.

https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Slice-1
https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Slice-10


Ignore the commented code below:
I've modified the model converter to do this directly
https://github.com/onnx/onnx/pull/3055
'''

# def add_initializer(model, tensor):
#     model.graph.initializer.extend([tensor])

# def slice(model):
#     for node in model.graph.node:
#         if node.op_type == "Slice" and len(node.attribute) > 0:
#             assert node.name, "Node has no name. Todo generate a unique tensor name"
#             axes = None
#             ends = None
#             starts = None
#             for i in node.attribute:
#                 if i.name == "axes":
#                     axes = i.ints
#                 if i.name == "ends":
#                     ends = i.ints
#                 if i.name == "starts":
#                     starts = i.ints
#             assert starts is not None and ends is not None
#             start_tensor = onnx.helper.make_tensor(node.name + "_starts", onnx_proto.TensorProto.INT32, [len(starts)], starts)
#             ends_tensor = onnx.helper.make_tensor(node.name + "_ends", onnx_proto.TensorProto.INT32, [len(ends)], ends)
#             add_initializer(model, start_tensor)
#             add_initializer(model, ends_tensor)
#             node.input.extend([node.name + "_starts", node.name + "_ends"])
            
#             if axes is not None:
#                 axes_tensor = onnx.helper.make_tensor(node.name + "_axes", onnx_proto.TensorProto.INT32, [len(axes)], axes)
#                 add_initializer(model, axes_tensor)
#                 node.input.extend([node.name + "_axes"])
#             del node.attribute[:]
            

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Model (< 10)")
    parser.add_argument("--output", required=True, help="Output Model (at 10)")
    args = parser.parse_args()
    return args


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


args = get_args()
model = onnx.load(args.input)
add_value_info_for_constants(model)
for init in model.graph.initializer:
    for value_info in model.graph.value_info:
        if init.name == value_info.name:
            model.graph.input.append(value_info)

# This requires a new enough version of the opset converter tool. Build onnx from source if needed 
converted_model = version_converter.convert_version(model, 11)
#slice(converted_model)
onnx.save(converted_model, args.output)