/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_transpose_attributes.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"
#include "pool_attributes.h"

namespace onnxruntime {
namespace systolic {

struct QLinearConvTransposeAttributes : public ConvTransposeAttributes {
    explicit QLinearConvTransposeAttributes(const OpNodeProtoHelper<ProtoHelperNodeContext>& info)
      : ConvTransposeAttributes(info) {
  }
  Status PrepareForCompute(OpKernelContext* context, bool has_bias, Prepare& p) const {
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* F = context->Input<Tensor>(3);
    const Tensor* B = has_bias ? context->Input<Tensor>(8) : nullptr;
    const TensorShape& input_shape = X->Shape().Slice(2);

    const int64_t num_input_channels = X->Shape()[1];
    const int64_t N = X->Shape()[0];
    const int64_t num_output_channels_multiplier = F->Shape()[1];
    const int64_t num_output_channels = num_output_channels_multiplier * group;

    // input validations
    if (group <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "group count is <= 0",
                             " group: ", group);
    }
    if (X->Shape().NumDimensions() != F->Shape().NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "X num_dims does not match W num_dims.",
                             " X: ", X->Shape().ToString().c_str(),
                             " W: ", F->Shape().ToString().c_str());
    }

    if (F->Shape()[0] != num_input_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "filter number not equal to input channel number.",
                             " filter_number: ", F->Shape()[0],
                             " num_input_channels: ", num_input_channels);
    }

    // it looks like num_output_channels is really k*group similar to how in the conv case
    // num_input_channels is k*group. hence removing the check for num_output_channels here.

    if (num_input_channels % group != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input channels is not divisible by group.",
                             " num_input_channels: ", num_input_channels,
                             " group: ", group);
    }

    std::vector<int64_t> kernel_shape;
    ORT_RETURN_IF_ERROR(ComputeKernelShape(F->Shape(), kernel_shape));

    std::vector<int64_t> local_output_padding(output_padding);
    if (local_output_padding.empty()) {
      local_output_padding.resize(kernel_shape.size(), 0);
    }
    std::vector<int64_t> local_pads;
    local_pads.reserve(2 * (input_shape.NumDimensions()));
    local_pads.assign(pads.begin(), pads.end());

    if (local_pads.empty()) {
      local_pads.resize(kernel_shape.size() * 2, 0);
    }
    std::vector<int64_t> local_dilations(dilations);
    if (local_dilations.empty()) {
      local_dilations.resize(kernel_shape.size(), 1);
    }
    std::vector<int64_t> local_strides(strides);
    if (local_strides.empty()) {
      local_strides.resize(kernel_shape.size(), 1);
    }

    std::vector<int64_t> Y_dims;

    ComputePadsAndOutputShape(input_shape, num_output_channels, kernel_shape,
                              local_strides, local_dilations, local_output_padding, N, &local_pads, &Y_dims);
    TensorShape Yshape(Y_dims);
    Tensor* Y = context->Output(0, Yshape);

    p.X = X;
    p.F = F;
    p.B = B;
    p.Y = Y;
    p.N = N;
    p.input_shape = std::move(input_shape);
    p.num_input_channels = num_input_channels;
    p.num_output_channels = num_output_channels;
    p.kernel_shape = std::move(kernel_shape);
    p.pads = std::move(local_pads);
    p.strides = std::move(local_strides);
    p.dilations = std::move(local_dilations);
    return Status::OK();
  }
};

template <typename T>
class QLinearConvTranspose : public OpKernel {
 public:
  QLinearConvTranspose(const OpKernelInfo& info) : OpKernel(info), conv_transpose_attrs_(info) {}

  Status Compute(OpKernelContext* context) const override;

 protected:
  Status DoConvTranspose(OpKernelContext* context) const;

 private:
  QLinearConvTransposeAttributes conv_transpose_attrs_;
};

} // namespace systolic
}  // namespace onnxruntime