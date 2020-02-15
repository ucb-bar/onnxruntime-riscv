#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace systolic {
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA(name) \
  ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ(Counter, name)         \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_ELSEWHERE(name, schema_func) \
  ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(__COUNTER__, name, schema_func)
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(Counter, name, schema_func) \
  ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func)
#define ONNX_SYSTOLIC_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func) \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(                \
      op_schema_register_once##name##Counter) ONNX_UNUSED =                     \
      schema_func(ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__))

void RegisterSystolicSchemas() {
  static const char* QLinearRelu_doc = R"DOC(
QLinearRelu is a Relu that works on int8
)DOC";

  ONNX_SYSTOLIC_OPERATOR_SCHEMA(QLinearRelu)
      .SinceVersion(1)
      .SetDoc(QLinearRelu_doc)
      .Input(0, "X", "1D input tensor", "T")
      .Output(0, "Y", "1D output tensor", "T")
      .TypeConstraint(
          "T",
          {"tensor(int8)", "tensor(uint8)"},
          "Constrain input and output types to int8/unint8 tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);
}

}  // namespace systolic
}  // namespace onnxruntime