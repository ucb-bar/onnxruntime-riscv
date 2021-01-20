// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/attention_cpu_base.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/qmath.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace systolic {


template <typename T>
class QAttention : public OpKernel, public contrib::AttentionCPUBase {
 public:
  QAttention(const OpKernelInfo& info) : OpKernel(info), contrib::AttentionCPUBase(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  BufferUniquePtr packed_weights_;
  size_t packed_weights_size_;
  TensorShape weight_shape_;
  bool weights_is_signed_;
};


}  // namespace systolic
}  // namespace onnxruntime