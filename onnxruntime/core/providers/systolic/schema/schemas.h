#pragma once

#include "core/graph/onnx_protobuf.h"
#include <cmath>
#include <iostream>

namespace ONNX_NAMESPACE {
void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
void globalPoolTypeShapeInference(ONNX_NAMESPACE::InferenceContext& ctx);
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace systolic {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::InferenceContext;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;
using ONNX_NAMESPACE::TypeProto;

#define ONNX_SYSTOLIC_OPERATOR_SCHEMA(name) \
  ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ(Counter, name)        \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_ELSEWHERE(name, schema_func) \
  ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(__COUNTER__, name, schema_func)
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(Counter, name, schema_func) \
  ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func)
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func) \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(                 \
      op_schema_register_once##name##Counter) ONNX_UNUSED =                      \
      schema_func(ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__))

void nhwcConvPoolShapeInference(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx,
    int input2Idx) {
  // we need the first input shape for this inference.
  if (!hasInputShape(ctx, input1Idx)) {
    return;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !hasInputShape(ctx, input2Idx)) {
    return;
  }

  auto input_shape = ctx.getInputType(input1Idx)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have atleast 2 dimensions");
  }

  if (input_shape.dim_size() != 4) {
    fail_shape_inference("More than 4 input dims to qlinearconv_nhwc");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // Reshape input to NCHW format for shape inference logic
  int input_shape_N = input_shape.dim(0).dim_value();
  int input_shape_H = input_shape.dim(1).dim_value();
  int input_shape_W = input_shape.dim(2).dim_value();
  int input_shape_C = input_shape.dim(3).dim_value();

  int input_shape_nchw_form[4] = {input_shape_N, input_shape_C, input_shape_H, input_shape_W};

  // Only MaxPool and Conv support dilation. For
  // simplicity of the code, we just treat the rest of them as having all-1s
  // dilation.
  std::vector<int64_t> dilations;
  if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else if (require_kernel_shape) {
    fail_shape_inference("Attribute kernel_shape must be specified");
  } else {
    auto second_input_shape =
        ctx.getInputType(input2Idx)->tensor_type().shape();
    if (second_input_shape.dim_size() != 4) {
      fail_shape_inference("Not 4 dimensions for weights of qlinearconv_nhwc");
    }
    int seoncd_input_M = second_input_shape.dim(0).dim_value();
    int second_input_kH = second_input_shape.dim(1).dim_value();
    int second_input_kW = second_input_shape.dim(2).dim_value();
    int second_input_C_by_group = second_input_shape.dim(3).dim_value();

    int second_input_shape_nchw_form[4] = {seoncd_input_M, second_input_C_by_group, second_input_kH, second_input_kW};

    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        fail_shape_inference("Missing dim for qlinearconv_nhwc");
      }
      kernel_shape.push_back(second_input_shape_nchw_form[i]);
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          assert(input_shape.dim(2 + i).has_dim_value() && 2 + i < 4 && "Accessing beyond vector size");
          residual = input_shape_nchw_form[2 + i];
          while (residual >= stride) {
            residual -= stride;
          }
        }
        int64_t total_pad = residual == 0 ? effective_kernel_shape[i] - stride : effective_kernel_shape[i] - residual;
        if (total_pad < 0)
          total_pad = 0;
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  auto output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    *output_shape->add_dim() = input_shape.dim(0);
    *output_shape->add_dim() = input_shape.dim(3);
  } else {
    *output_shape->add_dim() = input_shape.dim(0);
    auto& second_input_shape = getInputShape(ctx, input2Idx);
    if (second_input_shape.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    *output_shape->add_dim() = second_input_shape.dim(0);
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = output_shape->add_dim();
    assert(input_shape.dim(2 + i).has_dim_value() && 2 + i < 4 && "Overflow at 177");

    // how big is the input, including padding
    int64_t effective_input_size = input_shape_nchw_form[2 + i];
    effective_input_size += pads[i];
    effective_input_size += pads[i + kernel_shape_size];

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions;

    if (ceil_mode == 1)
      strided_kernel_positions = (int64_t)(std::ceil(
          (effective_input_size - effective_kernel_shape[i]) / float(strides[i])));
    else
      strided_kernel_positions =
          (effective_input_size - effective_kernel_shape[i]) / strides[i];

    // add in the initial position
    newdim->set_dim_value(1 + strided_kernel_positions);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto second_output_shape =
        ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }

  if (output_shape->dim_size() != 4) {
    fail_shape_inference("More than 4 output dimensions for qlinearconv_nhwc");
  }

  int output_shape_N = output_shape->dim(0).dim_value();
  int output_shape_C = output_shape->dim(1).dim_value();
  int output_shape_H = output_shape->dim(2).dim_value();
  int output_shape_W = output_shape->dim(3).dim_value();

  // Reshape output format back to NHWC format
  output_shape->mutable_dim(0)->set_dim_value(output_shape_N);
  output_shape->mutable_dim(1)->set_dim_value(output_shape_H);
  output_shape->mutable_dim(2)->set_dim_value(output_shape_W);
  output_shape->mutable_dim(3)->set_dim_value(output_shape_C);
}

void nhwcConvPoolShapeInference(InferenceContext&
                                    ctx) {
  auto x_type = ctx.getInputType(0);
  auto w_type = ctx.getInputType(3);
  if (nullptr == x_type || nullptr == w_type ||
      x_type->value_case() != TypeProto::kTensorType ||
      w_type->value_case() != TypeProto::kTensorType) {
    fail_type_inference("inputs are expected to have tensor type.");
  }

  auto x_zero_point_type = ctx.getInputType(2);
  if (nullptr == x_zero_point_type ||
      x_zero_point_type->tensor_type().elem_type() !=
          x_type->tensor_type().elem_type()) {
    fail_type_inference(
        "input and zero_point pair is expected to have be same type.");
  }

  auto w_zero_point_type = ctx.getInputType(5);
  if (nullptr == w_zero_point_type ||
      w_zero_point_type->tensor_type().elem_type() !=
          w_type->tensor_type().elem_type()) {
    fail_type_inference(
        "weight and zero_point pair is expected to have same type.");
  }

  propagateElemTypeFromInputToOutput(ctx, 7, 0);

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  nhwcConvPoolShapeInference(ctx, true, false, 0, 3);
}

void RegisterSystolicSchemas() {
  ONNX_SYSTOLIC_OPERATOR_SCHEMA(QLinearRelu)
      .SinceVersion(1)
      .SetDoc("A Relu that works on int8")
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input and output types to int8/unint8 tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_SYSTOLIC_OPERATOR_SCHEMA(QLinearConv_nhwc)
      .SinceVersion(10)
      .SetDoc("Internal node for NHWC layout optimization. Used with Systolic.")
      .Input(0, "x", "", "T1")
      .Input(1, "x_scale", "", "tensor(float)")
      .Input(2, "x_zero_point", "", "T1")
      .Input(3, "w", "Must be in funky group-wise pre-transposed format", "T2")
      .Input(4, "w_scale", "", "tensor(float)")
      .Input(5, "w_zero_point", "", "T2")
      .Input(6, "y_scale", "", "tensor(float)")
      .Input(7, "y_zero_point", "", "T3")
      .Input(8, "B", "", "T4", OpSchema::Optional)
      .Output(0, "y", "", "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input type to 8-bit integer tensor.")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain filter type to 8-bit integer tensor.")
      .TypeConstraint("T3", {"tensor(int8)", "tensor(uint8)"}, "Constrain output type to 8-bit integer tensor.")
      .TypeConstraint("T4", {"tensor(int32)"}, "Constrain bias type to 32-bit integer tensor.")
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
      .TypeAndShapeInferenceFunction(static_cast<void (*)(InferenceContext& ctx)>(nhwcConvPoolShapeInference));

  ONNX_SYSTOLIC_OPERATOR_SCHEMA(Fused_QLinearConv_Relu_nhwc)
      .SinceVersion(1)
      .SetDoc("Internal fused node for NHWC layout optimization. Used with Systolic.")
      .Input(0, "x", "", "T1")
      .Input(1, "x_scale", "", "tensor(float)")
      .Input(2, "x_zero_point", "", "T1")
      .Input(3, "w", "Must be in funky group-wise pre-transposed format", "T2")
      .Input(4, "w_scale", "", "tensor(float)")
      .Input(5, "w_zero_point", "", "T2")
      .Input(6, "y_scale", "", "tensor(float)")
      .Input(7, "y_zero_point", "", "T3")
      .Input(8, "B", "", "T4", OpSchema::Optional)
      .Output(0, "y", "", "T3")
      .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input type to 8-bit integer tensor.")
      .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain filter type to 8-bit integer tensor.")
      .TypeConstraint("T3", {"tensor(int8)", "tensor(uint8)"}, "Constrain output type to 8-bit integer tensor.")
      .TypeConstraint("T4", {"tensor(int32)"}, "Constrain bias type to 32-bit integer tensor.")
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
      .TypeAndShapeInferenceFunction(static_cast<void (*)(InferenceContext& ctx)>(nhwcConvPoolShapeInference));
}

}  // namespace systolic
}  // namespace onnxruntime