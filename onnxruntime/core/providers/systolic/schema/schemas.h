#pragma once

#include "core/graph/onnx_protobuf.h"
#include <cmath>
#include <iostream>
#include <vector>
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

std::vector<int64_t> nhwcConvPoolShapeInference(
    const ONNX_NAMESPACE::TensorShapeProto* in1_shape,
    const ONNX_NAMESPACE::TensorShapeProto* in2_shape,
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
    return {};
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !in2_shape) {
    return {};
  }

  auto input_shape = *in1_shape;
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have atleast 2 dimensions");
  }

  if (input_shape.dim_size() != 4) {
    fail_shape_inference("More than 4 input dims to qlinearconv_nhwc");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  for (int i = 0; i < 4; i++) {
    // Bail if it isn't statically known
    if (!input_shape.dim(i).has_dim_value()) {
      return {};
    }
  }

  // Reshape input to NCHW format for shape inference logic
  int input_shape_N = input_shape.dim(0).dim_value();
  int input_shape_H = input_shape.dim(1).dim_value();
  int input_shape_W = input_shape.dim(2).dim_value();
  int input_shape_C = input_shape.dim(3).dim_value();

  int input_shape_nchw_form[4] = {input_shape_N, input_shape_C, input_shape_H, input_shape_W};

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
    auto second_input_shape = *in2_shape;
    if (second_input_shape.dim_size() != 4) {
      fail_shape_inference("Not 4 dimensions for weights of qlinearconv_nhwc");
    }
    for (int i = 0; i < 4; i++) {
      // Bail if it isn't statically known
      if (!second_input_shape.dim(i).has_dim_value()) {
        return {};
      }
    }
    int second_input_kH = second_input_shape.dim(0).dim_value();
    int second_input_kW = second_input_shape.dim(1).dim_value();
    int second_input_C_by_group = second_input_shape.dim(2).dim_value();
    int second_input_M = second_input_shape.dim(3).dim_value();

    int second_input_shape_oihw_form[4] = {second_input_M, second_input_C_by_group, second_input_kH, second_input_kW};

    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i - 2).has_dim_value()) {
        fail_shape_inference("Missing dim for qlinearconv_nhwc");
      }
      kernel_shape.push_back(second_input_shape_oihw_form[i]);
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

  std::vector<int> output_shape;

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    output_shape.push_back(input_shape.dim(0).dim_value());
    output_shape.push_back(input_shape.dim(3).dim_value());
  } else {
    output_shape.push_back(input_shape.dim(0).dim_value());
    auto second_input_shape = *in2_shape;
    if (second_input_shape.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    output_shape.push_back(second_input_shape.dim(3).dim_value());
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    assert(input_shape.dim(2 + i).has_dim_value() && 2 + i < 4 && "Overflow at 177");

    // how big is the input, including padding
    int64_t effective_input_size = input_shape_nchw_form[2 + i];
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
    output_shape.push_back(1 + strided_kernel_positions);
  }

  if (output_shape.size() != 4) {
    fail_shape_inference("More than 4 output dimensions for qlinearconv_nhwc");
  }

  int output_shape_N = output_shape[0];
  int output_shape_C = output_shape[1];
  int output_shape_H = output_shape[2];
  int output_shape_W = output_shape[3];

  return {output_shape_N, output_shape_H, output_shape_W, output_shape_C};
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

  std::vector<int64_t> conv_output_shape = nhwcConvPoolShapeInference(
      hasInputShape(ctx, input1Idx) ? &in_shape : nullptr,
      hasInputShape(ctx, input2Idx) ? &in2_shape : nullptr,
      dilations, strides, pads, kernel_shape, auto_pad,
      /*ceil_mode= */ 0, /*use_dilation= */ true, /*require_kernel_shape= */ false);

  if (conv_output_shape.size() > 0) {
    auto output_shape =
        ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    output_shape->clear_dim();
    for (int64_t dim : conv_output_shape) {
      output_shape->add_dim()->set_dim_value(dim);
    }

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

      auto immutable_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->shape();
      std::vector<int64_t> pool_output_shape = nhwcConvPoolShapeInference(
          &immutable_output_shape, /*in2_shape= */ nullptr,
          pool_dilations, pool_strides, pool_pads, pool_kernel_shape, pool_auto_pad,
          /*ceil_mode= */ 0, /*use_dilation= */ false, /*require_kernel_shape= */ true);

      // dump_vector(conv_output_shape, "Shape after conv");
      // dump_vector(pool_output_shape, "Shape after pool");

      output_shape->clear_dim();
      for (int64_t dim : pool_output_shape) {
        output_shape->add_dim()->set_dim_value(dim);
      }
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
        //fprintf(stderr, "CALLED INTO SHAPE INFERENCE\n");
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

        auto pool_output_shape = nhwcConvPoolShapeInference(
            hasInputShape(ctx, 0) ? &in_shape : nullptr,
            nullptr,
            dilations, strides, pads, kernel_shape, auto_pad,
            ceil_mode, /*use_dilation= */ true, /*require_kernel_shape= */ true);
        //dump_vector(pool_output_shape, "Pool output shape");
        auto output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        output_shape->clear_dim();
        for (int64_t dim : pool_output_shape) {
          output_shape->add_dim()->set_dim_value(dim);
        }
      });
}

}  // namespace systolic
}  // namespace onnxruntime