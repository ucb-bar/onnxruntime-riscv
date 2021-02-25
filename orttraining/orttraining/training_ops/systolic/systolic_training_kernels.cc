// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/systolic/systolic_fwd.h"
#include "core/framework/kernel_registry.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace systolic {

#ifdef SYSTOLIC_FP32
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 9, ConvGrad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 9, ConvGrad_nhwc);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 9, MaxPoolGrad_nhwc);
#endif

Status RegisterSystolicTrainingKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
#ifdef SYSTOLIC_FP32
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 9, ConvGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 9, ConvGrad_nhwc)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSystolicExecutionProvider, kOnnxDomain, 9, MaxPoolGrad_nhwc)>,
#endif
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime
