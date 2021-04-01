#pragma once

#include "core/graph/onnx_protobuf.h"
#include <cmath>
#include <iostream>
#include <vector>
#include "core/common/optional.h"
#include "core/framework/utils.h"

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
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::TensorShapeProto;
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

optional<TensorShapeProto> nhwcConvPoolShapeInference(
    const TensorShapeProto* in1_shape,
    const TensorShapeProto* in2_shape,
    std::vector<int64_t>& dilations,
    std::vector<int64_t>& strides,
    std::vector<int64_t>& pads,
    std::vector<int64_t>& kernel_shape,
    const std::string& auto_pad,
    int ceil_mode,
    bool use_dilation,
    bool require_kernel_shape /*whether we need the kernel_shape attr (for pool) */) {
  // we need the first input shape for this inference.
  if (!in1_shape) {
    // This can happen if shape inference function is not defined for an upstream node
    // E.g. for fused nodes ORT currently does not apply any shape inference
    fprintf(stderr, "Warning: input shape to nhwc convpool shape inference not found.\n");
    return nullopt;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !in2_shape) {
    //fprintf(stderr, "Returning null since in2 shape not provided. Bad model?\n");
    return nullopt;
  }

  const TensorShapeProto& input_shape = *in1_shape;
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have atleast 2 dimensions");
  }

  if (input_shape.dim_size() != 4) {
    fail_shape_inference("More than 4 input dims to qlinearconv_nhwc");
  }

  // The input is given to us in NHWC format
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // We convert the input shape to NCHW format for inference purposes
  TensorShapeProto input_shape_nchw_form;
  *input_shape_nchw_form.add_dim() = input_shape.dim(0);
  *input_shape_nchw_form.add_dim() = input_shape.dim(3);
  *input_shape_nchw_form.add_dim() = input_shape.dim(1);
  *input_shape_nchw_form.add_dim() = input_shape.dim(2);

  // Our weights are given to us in HWIO form. We fill this in later
  TensorShapeProto second_input_shape_oihw_form;
  if (in2_shape && in2_shape->dim_size() == 4) {
    *second_input_shape_oihw_form.add_dim() = in2_shape->dim(3);
    *second_input_shape_oihw_form.add_dim() = in2_shape->dim(2);
    *second_input_shape_oihw_form.add_dim() = in2_shape->dim(0);
    *second_input_shape_oihw_form.add_dim() = in2_shape->dim(1);
  }

  // Only MaxPool and Conv support dilation. For
  // simplicity of the code, we just treat the rest of them as having all-1s
  // dilation.
  if (use_dilation && dilations.size() != 0) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  if (strides.size() != 0) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  if (kernel_shape.size() != 0) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else if (require_kernel_shape) {
    fail_shape_inference("Attribute kernel_shape must be specified");
  } else {
    for (int i = 2; i < second_input_shape_oihw_form.dim_size(); ++i) {
      if (!second_input_shape_oihw_form.dim(i).has_dim_value()) {
        return nullopt;
      }
      kernel_shape.push_back(second_input_shape_oihw_form.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  if (pads.size() != 0) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    if (auto_pad != "VALID") {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          if (!input_shape_nchw_form.dim(2 + i).has_dim_value()) {
            continue;
          }
          residual = input_shape_nchw_form.dim(2 + i).dim_value();
          while (residual >= stride) {
            residual -= stride;
          }
        }
        int64_t total_pad = residual == 0 ? effective_kernel_shape[i] - stride : effective_kernel_shape[i] - residual;
        if (total_pad < 0)
          total_pad = 0;
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  TensorShapeProto output_shape;

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    *output_shape.add_dim() = input_shape_nchw_form.dim(0);
    *output_shape.add_dim() = input_shape_nchw_form.dim(1);
  } else {
    *output_shape.add_dim() = input_shape_nchw_form.dim(0);
    if (second_input_shape_oihw_form.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    *output_shape.add_dim() = second_input_shape_oihw_form.dim(0);
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = output_shape.add_dim();
    if (!input_shape_nchw_form.dim(2 + i).has_dim_value()) {
      continue;
    }

    // how big is the input, including padding
    int64_t effective_input_size = input_shape_nchw_form.dim(2 + i).dim_value();
    effective_input_size += pads[i];
    effective_input_size += pads[i + kernel_shape_size];

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

  if (output_shape.dim_size() != 4) {
    fail_shape_inference("More than 4 output dimensions for qlinearconv_nhwc");
  }

  TensorShapeProto output_shape_nhwc;
  *output_shape_nhwc.add_dim() = output_shape.dim(0);
  *output_shape_nhwc.add_dim() = output_shape.dim(2);
  *output_shape_nhwc.add_dim() = output_shape.dim(3);
  *output_shape_nhwc.add_dim() = output_shape.dim(1);
  return optional<TensorShapeProto>(output_shape_nhwc);
}

void dump_vector(const std::vector<int64_t>& v, const std::string& title) {
  printf("Dumping %s: ", title.c_str());
  for (auto i : v) {
    printf("%zd ", i);
  }
  printf("\n");
}

void nhwcConvPoolShapeInference(InferenceContext& ctx, int x_idx, int w_idx) {
  int input1Idx = x_idx;
  int input2Idx = w_idx;

  auto in_shape = ctx.getInputType(input1Idx)->tensor_type().shape();
  auto in2_shape = ctx.getInputType(input2Idx)->tensor_type().shape();

  std::vector<int64_t> dilations;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> kernel_shape;
  getRepeatedAttribute(ctx, "dilations", dilations);
  getRepeatedAttribute(ctx, "strides", strides);
  getRepeatedAttribute(ctx, "pads", pads);
  getRepeatedAttribute(ctx, "kernel_shape", kernel_shape);

  const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
  std::string auto_pad = auto_pad_attr ? auto_pad_attr->s() : "NOTSET";

  optional<TensorShapeProto> conv_output_shape = nhwcConvPoolShapeInference(
      hasInputShape(ctx, input1Idx) ? &in_shape : nullptr,
      hasInputShape(ctx, input2Idx) ? &in2_shape : nullptr,
      dilations, strides, pads, kernel_shape, auto_pad,
      /*ceil_mode= */ 0, /*use_dilation= */ true, /*require_kernel_shape= */ false);

  if (conv_output_shape) {
    const auto* has_maxpool_attr = ctx.getAttribute("maxpool");
    if (has_maxpool_attr && has_maxpool_attr->i() == 1) {
      //printf("Has pool attribute");
      std::vector<int64_t> pool_dilations;
      std::vector<int64_t> pool_strides;
      std::vector<int64_t> pool_pads;
      std::vector<int64_t> pool_kernel_shape;
      getRepeatedAttribute(ctx, "pool_dilations", pool_dilations);
      getRepeatedAttribute(ctx, "pool_strides", pool_strides);
      getRepeatedAttribute(ctx, "pool_pads", pool_pads);
      getRepeatedAttribute(ctx, "pool_kernel_shape", pool_kernel_shape);

      // dump_vector(pool_dilations, "pool_dilations");
      // dump_vector(pool_strides, "pool_strides");
      // dump_vector(pool_pads, "pool_pads");
      // dump_vector(pool_kernel_shape, "pool_kernel_shape");

      const auto* pool_auto_pad_attr = ctx.getAttribute("pool_auto_pad");
      std::string pool_auto_pad = pool_auto_pad_attr ? pool_auto_pad_attr->s() : "NOTSET";

      optional<TensorShapeProto> pool_output_shape = nhwcConvPoolShapeInference(
          &*(conv_output_shape), /*in2_shape= */ nullptr,
          pool_dilations, pool_strides, pool_pads, pool_kernel_shape, pool_auto_pad,
          /*ceil_mode= */ 0, /*use_dilation= */ false, /*require_kernel_shape= */ true);

      // dump_vector(conv_output_shape, "Shape after conv");
      // dump_vector(pool_output_shape, "Shape after pool");
      if (pool_output_shape.has_value()) {
        ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->CopyFrom(pool_output_shape.value());
      }
    } else {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->CopyFrom(conv_output_shape.value());
    }
  }
}

void RegisterSystolicTrainingSchemas() {
  ONNX_SYSTOLIC_OPERATOR_SCHEMA(ConvGrad_nhwc)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Input(2, "W", "Weight tensor", "T")
      .Output(0, "dX", "Gradient of input X", "T", OpSchema::Optional)
      .Output(1, "dW", "Gradient of W", "T")
      .Output(2, "dB", "Gradient of B", "T", OpSchema::Optional)
      .AllowUncheckedAttributes()
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_SYSTOLIC_OPERATOR_SCHEMA(MaxPoolGrad_nhwc)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output, Y", "T")
      .Input(1, "Indices", "Indices tensor from max pooling across the input tensor.", "I")
      .Output(0, "dX", "Gradient of input, X", "T")
      .AllowUncheckedAttributes()
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain index tensor to int64");
}

void RegisterSystolicSchemas() {
#ifdef ENABLE_TRAINING
  RegisterSystolicTrainingSchemas();
#endif

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
      .SetDoc("Internal node for NHWC layout optimization. Also supports relu/maxpool Used with Systolic.")
      .Input(0, "x", "", "T1")
      .Input(1, "x_scale", "", "tensor(float)")
      .Input(2, "x_zero_point", "", "T1")
      .Input(3, "w", "Must be in HWIO format", "T2")
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

      .Attr("relu", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("maxpool", "", AttributeProto::INT, static_cast<int64_t>(0))

      .Attr("pool_auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("pool_ceil_mode", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("pool_dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pool_kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pool_pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pool_storage_order", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("pool_strides", "", AttributeProto::INTS, OPTIONAL_VALUE)

      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        int x_idx = 0;
        int w_idx = 3;

        propagateElemTypeFromInputToOutput(ctx, 7, 0);

        auto x_type = ctx.getInputType(x_idx);
        auto w_type = ctx.getInputType(w_idx);
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

        nhwcConvPoolShapeInference(ctx, x_idx, w_idx);
      });

  ONNX_SYSTOLIC_OPERATOR_SCHEMA(Conv_nhwc)
      .SinceVersion(1)
      .SetDoc("Internal node for NHWC layout optimization. Also supports relu/maxpool Used with Systolic.")
      .Input(0, "X", "", "T", OpSchema::Single, /*is_homogeneous= */ true, /*min_arity= */ 1, OpSchema::Differentiable)
      .Input(1, "W", "", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(2, "B", "", "T", OpSchema::Optional, true, 1, OpSchema::Differentiable)
      .Output(0, "Y", "", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")

      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))

      .Attr("relu", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("maxpool", "", AttributeProto::INT, static_cast<int64_t>(0))

      .Attr("pool_auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("pool_ceil_mode", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("pool_dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pool_kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pool_pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pool_storage_order", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("pool_strides", "", AttributeProto::INTS, OPTIONAL_VALUE)

      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        int x_idx = 0;
        int w_idx = 1;

        auto x_type = ctx.getInputType(x_idx);
        auto w_type = ctx.getInputType(w_idx);
        if (nullptr == x_type || nullptr == w_type ||
            x_type->value_case() != TypeProto::kTensorType ||
            w_type->value_case() != TypeProto::kTensorType) {
          fail_type_inference("inputs are expected to have tensor type.");
        }

        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        nhwcConvPoolShapeInference(ctx, x_idx, w_idx);
      });

  ONNX_SYSTOLIC_OPERATOR_SCHEMA(QLinearConvTranspose)
      .SinceVersion(10)
      .SetDoc("")
      .Input(0, "x", "", "T1")
      .Input(1, "x_scale", "", "tensor(float)")
      .Input(2, "x_zero_point", "", "T1")
      .Input(3, "w", "", "T2")
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

      .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("output_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("output_padding", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "Stride along each spatial axis.", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
      .TypeAndShapeInferenceFunction(
            [](InferenceContext& ctx) { 
              // TODO: We should also add a shape inference function for this
              // We can re-use the existing convtranspose shape inference
              // but we will need to account for the different input indices
              propagateElemTypeFromInputToOutput(ctx, 0, 0);
            });

  ONNX_SYSTOLIC_OPERATOR_SCHEMA(MaxPool_nhwc)
      .SinceVersion(1)
      .SetDoc("Internal node for NHWC training.")
      .Input(0, "X", "", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Output(0, "Y", "", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Output(1, "Y", "", "I", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .TypeConstraint("T", {"tensor(int8)", "tensor(uint8)", "tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
      .Attr("kernel_shape", "", AttributeProto::INTS)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("ceil_mode", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("storage_order", "", AttributeProto::INT, static_cast<int64_t>(0))
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
//        fprintf(stderr, "CALLED INTO MAXPOOL SHAPE INFERENCE\n");
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (ctx.getNumOutputs() > 1) {
          // MaxPool with two outputs case.
          auto output_type = ctx.getOutputType(1);
          if (output_type->value_case() == TypeProto::kTensorType ||
              output_type->value_case() == TypeProto::VALUE_NOT_SET) {
            output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          }
        }
        auto in_shape = ctx.getInputType(0)->tensor_type().shape();

        std::vector<int64_t> dilations;
        std::vector<int64_t> strides;
        std::vector<int64_t> pads;
        std::vector<int64_t> kernel_shape;
        getRepeatedAttribute(ctx, "dilations", dilations);
        getRepeatedAttribute(ctx, "strides", strides);
        getRepeatedAttribute(ctx, "pads", pads);
        getRepeatedAttribute(ctx, "kernel_shape", kernel_shape);
        auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

        const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
        std::string auto_pad = auto_pad_attr ? auto_pad_attr->s() : "NOTSET";

        optional<TensorShapeProto> pool_output_shape = nhwcConvPoolShapeInference(
            hasInputShape(ctx, 0) ? &in_shape : nullptr,
            nullptr,
            dilations, strides, pads, kernel_shape, auto_pad,
            ceil_mode, /*use_dilation= */ true, /*require_kernel_shape= */ true);

        if (pool_output_shape) {
          auto output_shape =
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          output_shape->CopyFrom(pool_output_shape.value());

          if (ctx.getNumOutputs() > 1) {
            auto second_output_shape =
                ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
            second_output_shape->CopyFrom(*output_shape);
          }
        }
      });
}

}  // namespace systolic
}  // namespace onnxruntime