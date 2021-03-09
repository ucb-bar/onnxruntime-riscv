// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class BatchNormalizationGrad final : public OpKernel {
 public:
  explicit BatchNormalizationGrad(const OpKernelInfo& info) : OpKernel(info)  {
    int64_t tmp_spatial;
    if (info.GetAttr<int64_t>("spatial", &tmp_spatial).IsOK()) {
      is_spatial_ = tmp_spatial;
    }
  }

  Status Compute(OpKernelContext* context) const override;

  private:
  int64_t is_spatial_ = 1;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BatchNormalizationGrad);
};

}  // namespace contrib
}  // namespace onnxruntime