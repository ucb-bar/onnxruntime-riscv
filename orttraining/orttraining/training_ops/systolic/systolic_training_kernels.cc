// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/systolic/systolic_fwd.h"
#include "core/framework/kernel_registry.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace systolic {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 9, ConvGrad);

Status RegisterSystolicTrainingKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 9, ConvGrad)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
