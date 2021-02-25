// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/nn/pool_attributes.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
class MaxPool_nhwc : public OpKernel {
 public:
  MaxPool_nhwc(const OpKernelInfo& info) : OpKernel(info), pool_attrs_(info, "MaxPool", 13) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  PoolAttributes pool_attrs_;
};

} // namespace systolic
}  // namespace onnxruntime
