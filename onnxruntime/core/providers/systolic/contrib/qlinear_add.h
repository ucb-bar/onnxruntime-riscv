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
  }

  Status Compute(OpKernelContext* context) const override;
  bool fused_relu_ = false;
};

template <typename T>
class FusedQLinearAddRelu : public QLinearAdd<T> {
 public:
  explicit FusedQLinearAddRelu(const OpKernelInfo& info) : QLinearAdd<T>(info) {
    this->fused_relu_ = true;
  }
};

} // namespace systolic
}  // namespace onnxruntime
