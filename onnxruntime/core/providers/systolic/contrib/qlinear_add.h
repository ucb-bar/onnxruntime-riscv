// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
class QLinearAdd : public OpKernel {
 public:
  explicit QLinearAdd(const OpKernelInfo& info) : OpKernel(info) {
    int64_t relu;
    auto status = info.GetAttr<int64_t>("relu", &relu);
    if (!status.IsOK()) {
      relu = 0;
    }
    fused_relu_ = (bool) relu;
  }

  Status Compute(OpKernelContext* context) const override;
  bool fused_relu_ = false;
};


} // namespace systolic
}  // namespace onnxruntime
