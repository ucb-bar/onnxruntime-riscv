// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"
#include "pool_attributes.h"

namespace onnxruntime {
namespace systolic {

class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    int64_t relu;
    if (info.GetAttr<int64_t>("relu", &relu).IsOK()) {
      fused_relu_ = (bool) relu;
    }
  }

  Status Compute(OpKernelContext* context) const override;
  ConvAttributes conv_attrs_;
  bool fused_relu_ = 0;
};

class QLinearConv_nhwc : public OpKernel {
 public:
  explicit QLinearConv_nhwc(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info), pool_attrs_(info) {
    int64_t relu;
    if (info.GetAttr<int64_t>("relu", &relu).IsOK()) {
      fused_relu_ = (bool) relu;
    }
  }

  Status Compute(OpKernelContext* context) const override;
  ConvAttributes conv_attrs_;

  // pool_attrs_ has fused_pool attribute
  PoolAttributes pool_attrs_;
  bool fused_relu_ = 0;
};

} // namespace systolic
}  // namespace onnxruntime
