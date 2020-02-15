// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace systolic {

template <typename T1>
class QLinearRelu final : public OpKernel {
 public:
  QLinearRelu(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

};

} // namespace systolic
}  // namespace onnxruntime
