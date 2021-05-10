// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"
#include "conv_pool_helper.h"

#ifdef SYSTOLIC_FP32

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    11,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1, 10,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv_nhwc,
    kOnnxDomain,
    1, 11,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv_nhwc<float>);

template <typename T>
Status Conv_nhwc<T>::Compute(OpKernelContext* context) const {
  profiling::Profiler& profiler = static_cast<OpKernelContextInternal*>(context)->GetProfiler();
  bool profiling_enabled = profiler.IsEnabled();

  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* B = nullptr;
  if (num_inputs == 3) {
    B = context->Input<Tensor>(2);
  }

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[3];
  const int64_t M = W->Shape()[3];
  ORT_RETURN_IF_ERROR(ValidateConvInputShapeNHWC(X, W, conv_attrs_.group));
  ORT_ENFORCE(B == nullptr || B->Shape().NumDimensions() == 1, "Bias is not 1D");
  ORT_ENFORCE(B == nullptr || B->Shape().Size() == M, "1D Bias does not match M");

  std::vector<int64_t> kernel_shape;
  TensorShape oihw_w_shape = {W->Shape()[3], W->Shape()[2], W->Shape()[0], W->Shape()[1]};
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(oihw_w_shape, kernel_shape));

  const size_t kernel_rank = kernel_shape.size();
  ORT_ENFORCE(kernel_rank == 2, "NHWC cannot handle kernel rank other than 2 atm");

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims_nchw({N, M});
  TensorShape input_shape = X->Shape().Slice(1, 3);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims_nchw));
  std::vector<int64_t> Y_dims = {Y_dims_nchw[0], Y_dims_nchw[2], Y_dims_nchw[3], Y_dims_nchw[1]};
  auto Y_dims_shape = TensorShape(Y_dims);

  Tensor* Y = context->Output(0, Y_dims_shape);

  // If we can run on Systolic, do so!
  if (TryConvOnSystolic<float, float>(
          static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode(),
          dilations,
          pads, strides, conv_attrs_.group, X, W, B, Y,
          Y_dims_shape, Y_dims_shape,
          fused_relu_, /*pool_attrs= */ nullptr, /*real_multiplier=*/ 1)) {
    return Status::OK();
  }

  // Else we run on CPU, and have to allocate temp buffer for pre-pooling if needed
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  TensorShape output_shape = Y->Shape().Slice(1, 3);

  // fprintf(stderr, "INPUT SHAPE %s\n", input_shape.ToString().c_str());
  // fprintf(stderr, "KERNEL SHAPE %s\n", W->Shape().ToString().c_str());
  // fprintf(stderr, "OUTPUT SHAPE %s\n", Y->Shape().ToString().c_str());

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C * input_image_size;
  const int64_t Y_offset = (Y->Shape().Size() / Y->Shape()[0]);
  const int64_t B_offset = static_cast<int>(M / conv_attrs_.group);
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = C * output_image_size * kernel_size;

  // The col buffer is stored in HWC order as well - the height and width, and
  // kernel_dim.

  BufferUniquePtr col_buffer;

  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    auto col_data = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
  } else {
#ifndef FOR_FIRESIM
    printf("1x1 case!\n");
#endif
  }

  auto* col_buffer_data = static_cast<float*>(col_buffer.get());

  const auto* Xdata = X->template Data<float>();
  const auto* Wdata = W->template Data<float>();
  const auto* Bdata = B != nullptr ? B->template Data<float>() : nullptr;
  auto* Ydata = Y->template MutableData<float>();

  for (int image_id = 0; image_id < N; ++image_id) {
    TimePoint start_time;
    if (profiling_enabled) {
      start_time = profiler.Now();
    }
    // We use a version of im2col that does all groups at once
    // Whereas official onnxruntime optimization (CPU kernel) has a version
    // that operates at a per-group level
    // IF one were to parallelize across multiple cores, you could use that
    // Refer to the CPU QLinearConv impl. to see how that works
    if (col_buffer_data != nullptr) {
      Im2Col_NHWC(
          Xdata,
          C,
          input_shape[0],
          input_shape[1],
          kernel_shape[0],
          kernel_shape[1],
          dilations[0],
          dilations[1],
          pads[0],
          pads[1],
          pads[2],
          pads[3],
          strides[0],
          strides[1],
          col_buffer_data,
          conv_attrs_.group,
          (float) 0.0);

      if (profiling_enabled) {
        profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                       Node().Name() + "_kernel_nhwc_im2col_time",
                                       start_time,
                                       {{"op_name", KernelDef().OpName()},
                                        {"sub_action", "im2col"},
                                        {"provider", KernelDef().Provider()}});
        start_time = profiler.Now();
      }
    }

    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      const float* weight_base = Wdata + group_id * static_cast<int>(M / conv_attrs_.group);
      SystolicMultiply(static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode(),
                       /*relu= */ fused_relu_,
                       static_cast<int>(output_image_size),
                       static_cast<int>(M / conv_attrs_.group),
                       static_cast<int>(kernel_dim),
                       (col_buffer_data == nullptr ? Xdata : col_buffer_data) + group_id * static_cast<int>(kernel_dim), conv_attrs_.group * static_cast<int>(kernel_dim),
                       weight_base, static_cast<int>(M),
                       Ydata + group_id * static_cast<int>(M / conv_attrs_.group), static_cast<int>(M),
                       /*multiplier= */ (float) 1.0,
                       Bdata != nullptr ? Bdata + group_id * B_offset : nullptr, static_cast<int>(M / conv_attrs_.group),
                       /*repeating_bias= */ true);

      if (profiling_enabled) {
        std::string dimension_string;
        dimension_string = std::to_string(static_cast<int>(M / conv_attrs_.group)) +
                           ", " + std::to_string(static_cast<int>(output_image_size)) + ", " +
                           std::to_string(static_cast<int>(kernel_dim));
        profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                       Node().Name() + "_kernel_matmul_time",
                                       start_time,
                                       {{"op_name", KernelDef().OpName()},
                                        {"sub_action", "matmul"},
                                        {"relu_fused", fused_relu_ ? "yes" : "no"},
                                        {"dimensions", dimension_string},
                                        {"provider", KernelDef().Provider()}});
        start_time = profiler.Now();
      }
    }

    Xdata += X_offset;
    Ydata += Y_offset;
  }

  return Status::OK();
}

template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);  // optional. nullptr if not provided
  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims({N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = context->Output(0, Y_dims);
  TensorShape output_shape = Y->Shape().Slice(2);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  const size_t kernel_rank = kernel_shape.size();

  BufferUniquePtr col_buffer;
  std::vector<int64_t> col_buffer_shape;

  // Pointwise convolutions can use the original input tensor in place,
  // otherwise a temporary buffer is required for the im2col transform.
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(T)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));

    if (kernel_rank != 2) {
      const auto& output_dims = output_shape.GetDims();
      col_buffer_shape.reserve(1 + output_dims.size());
      col_buffer_shape.push_back(kernel_dim);
      col_buffer_shape.insert(col_buffer_shape.end(), output_dims.begin(), output_dims.end());
    }
  }

  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  const T* Xdata = X->template Data<T>();
  T* Ydata = Y->template MutableData<T>();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (col_buffer_data != nullptr) {
        if (kernel_rank == 2) {
          math::Im2col<T, StorageOrder::NCHW>()(
              Xdata + group_id * X_offset,
              C / conv_attrs_.group,
              input_shape[0],
              input_shape[1],
              kernel_shape[0],
              kernel_shape[1],
              dilations[0],
              dilations[1],
              pads[0],
              pads[1],
              pads[2],
              pads[3],
              strides[0],
              strides[1],
              col_buffer_data);
        } else {
          math::Im2col<T, StorageOrder::NCHW>()(
              Xdata + group_id * X_offset,
              input_shape.GetDims().data(),
              output_shape.GetDims().data(),
              kernel_dim,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_shape.size()),
              col_buffer_data);
        }
      }

      SystolicMultiply(static_cast<const SystolicExecutionProvider*>(
                        this->Info().GetExecutionProvider())->GetAcceleratorMode(),
                      /* relu= */ false,
                      static_cast<int>(M / conv_attrs_.group),
                      static_cast<int>(output_image_size),
                      static_cast<int>(kernel_dim),
                      W->template Data<T>() + group_id * W_offset,
                      col_buffer_data == nullptr ? Xdata + group_id * X_offset : col_buffer_data,
                      Ydata + group_id * Y_offset,
                      /*real_multiplier=*/ 1, /* bias= */nullptr);
    }

    if (B != nullptr) {
      auto Ymatrix = EigenMatrixMap<T>(Ydata, output_image_size, M);
      auto Bvec = ConstEigenVectorMap<T>(B->template Data<T>(), M);
      Ymatrix.rowwise() += Bvec.transpose();
    }

    Xdata += X_offset * conv_attrs_.group;
    Ydata += Y_offset * conv_attrs_.group;
  }

  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime

#endif