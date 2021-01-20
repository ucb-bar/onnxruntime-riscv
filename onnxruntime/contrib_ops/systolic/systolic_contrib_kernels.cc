// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/systolic/systolic_fwd.h"
#include "core/framework/kernel_registry.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace systolic {

#ifdef SYSTOLIC_INT8
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kMSDomain, 1, float, QAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kMSDomain, 1, int8_t, QLinearAdd);
#endif

Status RegisterSystolicContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
#ifdef SYSTOLIC_INT8
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kMSDomain, 1, float, QAttention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kMSDomain, 1, int8_t, QLinearAdd)>,
#endif
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime
