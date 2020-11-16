// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace systolic {

class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    int64_t relu;
    auto status = info.GetAttr<int64_t>("relu", &relu);
    if (!status.IsOK()) {
      relu = 0;
    }
    fused_relu_ = (bool) relu;
  }

  Status Compute(OpKernelContext* context) const override;
  ConvAttributes conv_attrs_;
  bool fused_relu_ = false;
};

class QLinearConv_nhwc : public OpKernel {
 public:
  explicit QLinearConv_nhwc(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    int64_t relu;
    auto status = info.GetAttr<int64_t>("relu", &relu);
    if (!status.IsOK()) {
      relu = 0;
    }
    fused_relu_ = (bool) relu;
  }

  Status Compute(OpKernelContext* context) const override;
  ConvAttributes conv_attrs_;
  bool fused_relu_ = false;
};

} // namespace systolic
}  // namespace onnxruntime
