// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
class MaxPoolGrad_nhwc final : public OpKernel {
 public:
  explicit MaxPoolGrad_nhwc(const OpKernelInfo& info) : OpKernel(info) {
    auto& node = info.node();
    auto output_defs = node.OutputDefs();
    auto outputCount = output_defs.size();

    for (size_t outputIndex = 0; outputIndex < outputCount; outputIndex++) {
      output_tensor_shapes_.push_back({});
      if (!output_defs[outputIndex]->Exists())
        continue;

      auto shape = output_defs[outputIndex]->Shape();
      for (auto dim : shape->dim()) {
        output_tensor_shapes_[outputIndex].push_back(dim.dim_value());
      }
    }
    ORT_ENFORCE(!output_tensor_shapes_[0].empty());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MaxPoolGrad_nhwc);
  std::vector<VectorInt64> output_tensor_shapes_;
};

}  // namespace systolic
}  // namespace onnxruntime