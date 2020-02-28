// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace systolic {

template <StorageOrder STORAGE_ORDER>
class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;
  ConvAttributes conv_attrs_;
  bool fused_relu_ = false;
};

template <StorageOrder STORAGE_ORDER>
class FusedQLinearConvRelu : public QLinearConv<STORAGE_ORDER> {
 public:
  explicit FusedQLinearConvRelu(const OpKernelInfo& info) : QLinearConv<STORAGE_ORDER>(info) {
    this->fused_relu_ = true;
  }
};

} // namespace systolic
}  // namespace onnxruntime
