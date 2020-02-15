// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinearrelu.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/providers/systolic/helper/helper.h"

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Fused_QLinearConv_Relu,
    kOnnxDomain,
    1,
    int8_t,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    QLinearRelu<int8_t>);

template <>
Status QLinearRelu<int8_t>::Compute(OpKernelContext* ctx) const {
  ORT_UNUSED_PARAMETER(ctx);
  printf("Called into systolic!\n");
  return Status::OK();
}

} // namespace systolic
}  // namespace onnxruntime
