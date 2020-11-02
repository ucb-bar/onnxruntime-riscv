// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace systolic {

template <StorageOrder T>
class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;
  ConvAttributes conv_attrs_;
  bool fused_relu_ = false;
  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;

  bool has_reordered_w_ = false;
  BufferUniquePtr reordered_W_buffer_;
  TensorShape W_shape_;
};

template <StorageOrder T>
class FusedQLinearConvRelu : public QLinearConv<T> {
 public:
  explicit FusedQLinearConvRelu(const OpKernelInfo& info) : QLinearConv<T>(info) {
    this->fused_relu_ = true;
  }
};

} // namespace systolic
}  // namespace onnxruntime
