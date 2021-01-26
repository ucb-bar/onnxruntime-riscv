// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
class ConvGrad final : public OpKernel {
 public:
  explicit ConvGrad(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  ConvAttributes conv_attrs_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvGrad);
};

template <typename T>
class ConvGrad_nhwc final : public OpKernel {
 public:
  explicit ConvGrad_nhwc(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  ConvAttributes conv_attrs_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvGrad_nhwc);
};

}  // namespace systolic
}  // namespace onnxruntime