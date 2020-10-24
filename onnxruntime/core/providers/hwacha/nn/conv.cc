// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"
#include "core/providers/hwacha/hwacha_fwd.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/hwacha/hwacha_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace hwacha {

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    11,
    kHwachaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1, 10,
    kHwachaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  ORT_UNUSED_PARAMETER(context);
  ORT_ENFORCE(false, "This is a dummy operator");

  return Status::OK();
}

}  // namespace hwacha
}  // namespace onnxruntime