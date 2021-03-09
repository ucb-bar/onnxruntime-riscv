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

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include <safeint/SafeInt.hpp>

namespace onnxruntime {

template <typename T>
class BatchNorm : public OpKernel {
 public:
  explicit BatchNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info),
                                                           is_spatial_(op_kernel_info.GetAttrOrDefault<int64_t>("spatial", 1) == 1) {
    auto st = op_kernel_info.GetAttr<float>("epsilon", &epsilon_);
    ORT_ENFORCE(st.IsOK(), st.ErrorMessage());
    auto mt = op_kernel_info.GetAttr<float>("momentum", &momentum_);
    ORT_ENFORCE(mt.IsOK(), mt.ErrorMessage());
    // For opset 6-8, if spatial attribute exists, pick up the value (by default spatial == 1)
    // From opset 9 onwards, by default, only the spatial case (spatial == 1) is defined per spec

    // This scenario doesn't guarantee we're in training mode, but since ONNX schema lacks is_training attribute
    // We must make do with this
    is_train_ = OpKernel::Node().OutputDefs().size() == 5;
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    const auto* X = p_op_kernel_context->Input<Tensor>(0);
    const auto* scale = p_op_kernel_context->Input<Tensor>(1);
    const auto* B = p_op_kernel_context->Input<Tensor>(2);
    const auto* mean = p_op_kernel_context->Input<Tensor>(3);
    const auto* var = p_op_kernel_context->Input<Tensor>(4);

    ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, is_spatial_));

    const TensorShape& x_shape = X->Shape();
    Tensor* Y = p_op_kernel_context->Output(0, x_shape);

    const auto& dims_vec = x_shape.GetDims();
    const size_t N = dims_vec[0];
    const size_t C = dims_vec[1];  // assume NCHW as per the spec

    // calculate sample_size (per individual channel)
    size_t sample_size = 1;
    for (size_t i = 2; i < dims_vec.size(); ++i) {
      sample_size *= gsl::narrow<size_t>(dims_vec[i]);
    }

    // calculate sample_size (including all channels)
    size_t sample_size_incl_all_channels = sample_size * C;
    size_t scale_tensor_size = is_spatial_ ? C : sample_size_incl_all_channels;

    ConstEigenVectorArrayMap<T> scale_arr(scale->template Data<T>(), scale_tensor_size);
    ConstEigenVectorArrayMap<T> bias_arr(B->template Data<T>(), scale_tensor_size);

    // The saved mean corresponds to the mean from this batch
    auto* saved_mean = is_train_ ? p_op_kernel_context->Output(3, mean->Shape()) : nullptr;
    auto* saved_var = is_train_ ? p_op_kernel_context->Output(4, var->Shape()) : nullptr;

    // The running mean corresponds to the mean from all the batches
    // During inference this running mean is used as the mean for BN
    auto* running_mean = is_train_ ? p_op_kernel_context->Output(1, mean->Shape()) : nullptr;
    auto* running_var = is_train_ ? p_op_kernel_context->Output(2, var->Shape()) : nullptr;

    ConstEigenArrayMap<T> X_arr(X->template Data<T>(),
                                is_spatial_ ? sample_size : sample_size_incl_all_channels,
                                is_spatial_ ? N * C : N);

    if (is_train_) {
      EigenVectorArrayMap<T> saved_mean_arr(saved_mean->template MutableData<T>(), scale_tensor_size);
      EigenVectorArrayMap<T> saved_var_arr(saved_var->template MutableData<T>(), scale_tensor_size);
      saved_mean_arr.setZero();
      saved_var_arr.setZero();

      if (is_spatial_) {
        for (size_t nc = 0; nc < N * C; ++nc) {
          saved_mean_arr(nc % C) += X_arr.col(nc).sum();
        }
      } else {
        for (size_t n = 0; n < N; ++n) {
          saved_mean_arr.col(0) += X_arr.col(n).sum();
        }
      }

      saved_mean_arr /= N * (is_spatial_ ? sample_size : sample_size_incl_all_channels);
      if (is_spatial_) {
        for (size_t nc = 0; nc < N * C; ++nc) {
          saved_var_arr(nc % C) += (X_arr.col(nc) - saved_mean_arr(nc % C)).matrix().squaredNorm();
        }
      } else {
        for (size_t n = 0; n < N; ++n) {
          saved_var_arr.col(0) += (X_arr.col(n) - saved_mean_arr.col(0)).matrix().squaredNorm();
        }
      }
      saved_var_arr /= N * (is_spatial_ ? sample_size : sample_size_incl_all_channels);

      // Assume that running mean and variance are initialized properly in the model given to us
      // Because we alias it, we have the past history here
      EigenVectorArrayMap<T> running_mean_arr(
          running_mean->template MutableData<T>(), scale_tensor_size);
      EigenVectorArrayMap<T> running_var_arr(
          running_var->template MutableData<T>(), scale_tensor_size);
      running_mean_arr = running_mean_arr * momentum_ + saved_mean_arr * (1. - momentum_);
      running_var_arr = running_var_arr * momentum_ + saved_var_arr * (1. - momentum_);
    }

    // Regardless of training or testing, we will apply the estimated mean
    // and standard deviation to the input. For testing, they are
    // specified directly by the input, and for training, they are computed
    // by the op.
    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(var->Shape().Size());

    if (!is_train_) {
      ConstEigenVectorArrayMap<T> var_arr(var->template Data<T>(), scale_tensor_size);
      inv_std = (var_arr + epsilon_).sqrt().inverse();
    } else {
      // Note that, to be consistent with cudnn, we will actually output saved inverse std
      // This breaks the ONNX spec, but the existing cuda kernel for batchnormgrad relies on this behavior
      EigenVectorArrayMap<T> saved_inv_std(saved_var->template MutableData<T>(), scale_tensor_size);
      saved_inv_std = (saved_inv_std + epsilon_).inverse().sqrt();
      inv_std = saved_inv_std;
    }

    // If we're training, do batch normalization based on computation from this batch
    ConstEigenVectorArrayMap<T> mean_arr(
        !is_train_ ? mean->template Data<T>() : saved_mean->template Data<T>(), scale_tensor_size);

    // We can fuse the output computation as follows:
    //   ((x - est_mean) * (inv_var) * scale + bias
    // to
    //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
    Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
    Eigen::Array<T, Eigen::Dynamic, 1> new_bias = bias_arr - mean_arr * new_scale;
    EigenArrayMap<T> Y_arr(Y->template MutableData<T>(),
                           is_spatial_ ? sample_size : sample_size_incl_all_channels,
                           is_spatial_ ? N * C : N);

    if (is_spatial_) {  // spatial == 1
      for (size_t nc = 0; nc < N * C; ++nc) {
        Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
      }
    } else {  // spatial == 0
      for (size_t n = 0; n < N; ++n) {
        Y_arr.col(n) = X_arr.col(n) * new_scale.col(0) + new_bias.col(0);
      }
    }

    return Status::OK();
  }

 protected:
  float epsilon_;
  float momentum_;
  const bool is_spatial_;
  int64_t is_train_;
};
}  // namespace onnxruntime
