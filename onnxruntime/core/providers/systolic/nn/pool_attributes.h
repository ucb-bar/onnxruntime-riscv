// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace systolic {

// A helper struct holding attributes for Pool-family ops
struct PoolAttributes {
  PoolAttributes(const OpNodeProtoHelper<ProtoHelperNodeContext>& info) {
    int64_t maxpool;
    if (info.GetAttr<int64_t>("maxpool", &maxpool).IsOK()) {
      fused_pool = (bool) maxpool;
    }
    if (!fused_pool) {
      return;
    }

    ORT_ENFORCE(info.GetAttrs<int64_t>("pool_kernel_shape", kernel_shape).IsOK(),
                "No kernel shape is set.");
    ORT_ENFORCE(kernel_shape.size() == 2, "Pool must be 2x2");

    std::string auto_padding;
    ORT_ENFORCE(info.GetAttr<std::string>("pool_auto_pad", &auto_padding).IsOK());
    auto_pad = StringToAutoPadType(auto_padding);
    ORT_ENFORCE(auto_pad == AutoPadType::NOTSET);

    if (!info.GetAttrs<int64_t>("pool_pads", pads).IsOK() || pads.empty()) {
      pads.resize(kernel_shape.size() * 2, 0);
    }
    for (size_t i = 0; i < pads.size() / 2; i++) {
      ORT_ENFORCE(pads[i] == pads[pads.size() / 2 + i], "Cannot handle pads where begin != end");
    }

    if (!info.GetAttrs<int64_t>("pool_strides", strides).IsOK() || strides.empty()) {
      strides.resize(kernel_shape.size(), 1);
    }

    if (!info.GetAttr<int64_t>("pool_ceil_mode", &ceil_mode).IsOK()) {
      ceil_mode = 0;
    }
    ORT_ENFORCE(ceil_mode == 0);

    default_dilations = false;
    if (!info.GetAttrs<int64_t>("pool_dilations", dilations).IsOK() || dilations.empty()) {
      dilations.resize(kernel_shape.size(), 1);
      default_dilations = true;
    } else {
      default_dilations = std::all_of(dilations.begin(), dilations.end(), [](int64_t i) { return i == 1; });
    }
    ORT_ENFORCE(default_dilations == true, "Cannot handle non-1 dilation");

    ORT_ENFORCE(!info.GetAttr("storage_order", &storage_order).IsOK() || storage_order == 0);

    for (size_t dim = 0; dim < kernel_shape.size(); ++dim) {
      ORT_ENFORCE(kernel_shape[dim] > 0);
      ORT_ENFORCE(pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim],
                  "Pad should be smaller than kernel.");
    }

    ORT_ENFORCE(strides.size() == kernel_shape.size());
    ORT_ENFORCE(dilations.size() == kernel_shape.size(),
                "Dilations dimensions should match kernel shape");
  }

  int64_t storage_order{0};  // MaxPool_8 only. 0 is row major, and 1 is column major. Default is 0.
  int64_t ceil_mode{0};      // Introduced in MaxPool_10
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;  // Introduced in MaxPool_10
  // default_dilations is true if dilations is not set or all dilations are 1
  bool default_dilations;
  AutoPadType auto_pad;
  bool fused_pool{false};

  std::vector<int64_t> SetOutputSize_nhwc(const TensorShape& input_shape,
                                          int64_t output_channel,
                                          const std::vector<int64_t>& actual_pads) const {
    ORT_ENFORCE(input_shape.Size() > 0 || input_shape[0] == 0,
                "Invalid input shape. Only N can be zero. Got:", input_shape);

    TensorShape nchw_input_shape = {input_shape[0], input_shape[3], input_shape[1], input_shape[2]};
    std::vector<int64_t> output_dims;
    int64_t N = input_shape[0];
    InferOutputSize(nchw_input_shape.GetDims(), output_dims, actual_pads);

    output_dims.insert(output_dims.begin(), {N});
    output_dims.push_back(output_channel);

    return output_dims;
  }

  void InferOutputSize(const std::vector<int64_t>& input_dims,
                       std::vector<int64_t> &output_dims,
                       const std::vector<int64_t>& actual_pads) const {
    for (size_t dim = 0; dim < input_dims.size() - 2; ++dim) {
      int64_t dim_size = ComputeSizePadDilations(static_cast<int>(input_dims[dim + 2]),
                              strides[dim],
                              kernel_shape[dim],
                              &actual_pads.at(dim),
                              &actual_pads.at(input_dims.size() + dim - 2),
                              dilations[dim]);
      output_dims.push_back(dim_size);
    }
  }

  int64_t ComputeSizePadDilations(const int64_t in_size,
                          const int64_t stride,
                          const int64_t kernel,
                          const int64_t* pad_head,
                          const int64_t* pad_tail,
                          int64_t dilation) const {
    return ComputeOutputSize(in_size, stride, kernel, *pad_head + *pad_tail, dilation);
  }

  int64_t ComputeOutputSize(int64_t in_size,
                            int64_t stride,
                            int64_t kernel,
                            int64_t pad_needed,
                            int64_t dilation) const {
    return static_cast<int64_t>(static_cast<float>(in_size + pad_needed - dilation * (kernel - 1) - 1) / stride + 1);
  }
};

}  // namespace systolic
}  // namespace onnxruntime
