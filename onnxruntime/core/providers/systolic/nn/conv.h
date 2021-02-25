// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "pool_attributes.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ConvAttributes conv_attrs_;
  bool fused_relu_ = false;
};

template <typename T>
class Conv_nhwc : public OpKernel {
 public:
  Conv_nhwc(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ConvAttributes conv_attrs_;
  bool fused_relu_ = false;
};

} // namespace systolic
}  // namespace onnxruntime
