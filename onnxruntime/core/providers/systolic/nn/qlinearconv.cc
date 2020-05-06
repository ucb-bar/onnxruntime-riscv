// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinearconv.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    int8_t,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv<StorageOrder::NCHW>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv_nhwc,
    kOnnxDomain,
    10,
    int8_t,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv<StorageOrder::NHWC>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Fused_QLinearConv_Relu,
    kOnnxDomain,
    1,
    int8_t,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    FusedQLinearConvRelu<StorageOrder::NCHW>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Fused_QLinearConv_Relu_nhwc,
    kOnnxDomain,
    1,
    int8_t,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    FusedQLinearConvRelu<StorageOrder::NHWC>);

/**
 * Reference https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op_impl.h
 */
template <>
Status QLinearConv<StorageOrder::NHWC>::Compute(OpKernelContext* context) const {
  profiling::Profiler& profiler = static_cast<OpKernelContextInternal*>(context)->GetProfiler();
  bool profiling_enabled = profiler.IsEnabled();

  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(3);

  // validate offsets
  auto X_zero_point = context->Input<Tensor>(2);
  auto W_zero_point = context->Input<Tensor>(5);
  auto Y_zero_point = context->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_zero_point),
              "QLinearConv : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_zero_point),
              "QLinearConv : filter zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_zero_point),
              "QLinearConv : result zero point must be a scalar or 1D tensor of size 1");

  auto X_zero_point_value = *(X_zero_point->template Data<int8_t>());
  auto W_zero_point_value = *(W_zero_point->template Data<int8_t>());
  auto Y_zero_point_value = *(Y_zero_point->template Data<int8_t>());
  ORT_ENFORCE(X_zero_point_value == 0, "Systolic can only handle zero offset for input");
  ORT_ENFORCE(W_zero_point_value == 0, "Systolic can only handle zero offset for filter");
  ORT_ENFORCE(Y_zero_point_value == 0, "Systolic can only handle zero offset for result");

  // validate scale
  auto X_scale = context->Input<Tensor>(1);
  auto W_scale = context->Input<Tensor>(4);
  auto Y_scale = context->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
              "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_scale),
              "QLinearConv : filter scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
              "QLinearConv : result scale must be a scalar or 1D tensor of size 1");

  auto X_scale_value = *(X_scale->template Data<float>());
  auto W_scale_value = *(W_scale->template Data<float>());
  auto Y_scale_value = *(Y_scale->template Data<float>());

  ORT_ENFORCE(Y_scale_value != 0, "Y_scale_value cannot be 0");
  ORT_ENFORCE(W_scale_value != 0, "W_scale_value cannot be 0");
  ORT_ENFORCE(X_scale_value != 0, "X_scale_value cannot be 0");

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* B = nullptr;
  if (num_inputs == 9) {
    B = context->Input<Tensor>(8);
  }

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[3];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShapeNHWC(X, W));

  std::vector<int64_t> kernel_shape;
  TensorShape nchw_w_shape = {W->Shape()[0], W->Shape()[3], W->Shape()[1], W->Shape()[2]};
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(nchw_w_shape, kernel_shape));

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
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims_nchw));
  std::vector<int64_t> Y_dims = {Y_dims_nchw[0], Y_dims_nchw[2], Y_dims_nchw[3], Y_dims_nchw[1]};
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(1, 3);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

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
  
  const size_t kernel_rank = kernel_shape.size();
  ORT_ENFORCE(kernel_rank == 2, "NHWC cannot handle kernel rank other than 2 atm");

  // The col buffer is stored in HWC order as well - the height and width, and
  // kernel_dim.
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  
  BufferUniquePtr col_buffer;
  
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    auto col_data = alloc->Alloc(SafeInt<size_t>(sizeof(int8_t)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
  } else {
    //printf("1x1 case!\n");
  }

  auto* col_buffer_data = static_cast<int8_t*>(col_buffer.get());

  const float real_multiplier = (X_scale_value * W_scale_value) / Y_scale_value;
  unsigned int rounded_divisor = nearestPowerOfTwo(Y_scale_value / (X_scale_value * W_scale_value));

  const auto* Xdata = X->template Data<int8_t>();
  const auto* Wdata = W->template Data<int8_t>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->template MutableData<int8_t>();

  for (int image_id = 0; image_id < N; ++image_id) {
    TimePoint start_time;
    if (profiling_enabled) {
      start_time = profiler.StartTime();
    }
    if (col_buffer_data != nullptr) {
      math::Im2col<int8_t, StorageOrder::NHWC>()(
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
          X_zero_point_value);

      if (profiling_enabled) {
        profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                      Node().Name() + "_kernel_nhwc_im2col_time",
                                      start_time,
                                      {{"op_name", KernelDef().OpName()},
                                        {"sub_action", "im2col"},
                                        {"provider", KernelDef().Provider()}});
        start_time = profiler.StartTime();
      }
    }

    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      /* The weights we multiply by are given by the `M / groups` x `kernel_h * kernel_w * C / groups` matrix
       * starting at weights + group_id * (M / groups) * kernel_h * kernel_w * C / groups.
       * 
       * The quantization script will have already transposed this for us
       */
      const int8_t* weight_base = Wdata + group_id * static_cast<int>(M / conv_attrs_.group) * static_cast<int>(kernel_dim);
      // int8_t transposed[static_cast<int>(kernel_dim) * static_cast<int>(M / conv_attrs_.group)];
      // for (int i = 0; i < static_cast<int>(kernel_dim); i++) {
      //   for (int j = 0; j < static_cast<int>(M / conv_attrs_.group); j++) {
      //     transposed[i * static_cast<int>(M / conv_attrs_.group) + j] = weight_base[j * static_cast<int>(kernel_dim) + i];
      //   }
      // }

      /**
       * Note that when quantizing we will have transposed the weight matrix so that it holds the correct values
       */
      SystolicMultiplyi8i8_i8(static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode(),
                              /*relu= */ fused_relu_,
                              static_cast<int>(output_image_size),
                              static_cast<int>(M / conv_attrs_.group),
                              static_cast<int>(kernel_dim),
                              (col_buffer_data == nullptr ? Xdata : col_buffer_data) + group_id * static_cast<int>(kernel_dim), conv_attrs_.group * static_cast<int>(kernel_dim),
                              weight_base, static_cast<int>(M / conv_attrs_.group),
                              Ydata + group_id * static_cast<int>(M / conv_attrs_.group), static_cast<int>(M),
                              rounded_divisor, real_multiplier,
                              Bdata != nullptr ?  Bdata + group_id * B_offset : nullptr, static_cast<int>(M / conv_attrs_.group),
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
        start_time = profiler.StartTime();
      }

      // GemmlowpDebug(static_cast<int>(output_image_size),
      //               static_cast<int>(M / conv_attrs_.group),
      //               static_cast<int>(kernel_dim),
      //               col_buffer_data + group_id * static_cast<int>(kernel_dim), conv_attrs_.group * static_cast<int>(kernel_dim),
      //               transposed, static_cast<int>(M / conv_attrs_.group),
      //               Ydata + group_id * static_cast<int>(M / conv_attrs_.group), static_cast<int>(M),
      //               rounded_divisor,
      //               broadcast_bias.get(), static_cast<int>(M / conv_attrs_.group));
    }

    Xdata += X_offset;
    Ydata += Y_offset;
  }

  return Status::OK();
}

template <>
Status QLinearConv<StorageOrder::NCHW>::Compute(OpKernelContext* context) const {
  profiling::Profiler& profiler = static_cast<OpKernelContextInternal*>(context)->GetProfiler();
  bool profiling_enabled = profiler.IsEnabled();

  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(3);

  // validate offsets
  auto X_zero_point = context->Input<Tensor>(2);
  auto W_zero_point = context->Input<Tensor>(5);
  auto Y_zero_point = context->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_zero_point),
              "QLinearConv : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_zero_point),
              "QLinearConv : filter zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_zero_point),
              "QLinearConv : result zero point must be a scalar or 1D tensor of size 1");

  auto X_zero_point_value = *(X_zero_point->template Data<int8_t>());
  auto W_zero_point_value = *(W_zero_point->template Data<int8_t>());
  auto Y_zero_point_value = *(Y_zero_point->template Data<int8_t>());
  ORT_ENFORCE(X_zero_point_value == 0, "Systolic can only handle zero offset for input");
  ORT_ENFORCE(W_zero_point_value == 0, "Systolic can only handle zero offset for filter");
  ORT_ENFORCE(Y_zero_point_value == 0, "Systolic can only handle zero offset for result");

  // validate scale
  auto X_scale = context->Input<Tensor>(1);
  auto W_scale = context->Input<Tensor>(4);
  auto Y_scale = context->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
              "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_scale),
              "QLinearConv : filter scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
              "QLinearConv : result scale must be a scalar or 1D tensor of size 1");

  auto X_scale_value = *(X_scale->template Data<float>());
  auto W_scale_value = *(W_scale->template Data<float>());
  auto Y_scale_value = *(Y_scale->template Data<float>());
  ORT_ENFORCE(X_scale_value != 0, "X_scale_value cannot be 0");
  ORT_ENFORCE(W_scale_value != 0, "W_scale_value cannot be 0");
  ORT_ENFORCE(Y_scale_value != 0, "Y_scale_value cannot be 0");

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* B = nullptr;
  if (num_inputs == 9) {
    B = context->Input<Tensor>(8);
  }

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
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
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
  const int64_t B_offset = M / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  const size_t kernel_rank = kernel_shape.size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr col_buffer;
  std::vector<int64_t> col_buffer_shape;

  // Pointwise convolutions can use the original input tensor in place,
  // otherwise a temporary buffer is required for the im2col transform.
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(int8_t)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));

    if (kernel_rank != 2) {
      const auto& output_dims = output_shape.GetDims();
      col_buffer_shape.reserve(1 + output_dims.size());
      col_buffer_shape.push_back(kernel_dim);
      col_buffer_shape.insert(col_buffer_shape.end(), output_dims.begin(), output_dims.end());
    }
  } else {
    //printf("1x1 case!\n");
  }

  auto* col_buffer_data = static_cast<int8_t*>(col_buffer.get());

  const float real_multiplier = (X_scale_value * W_scale_value) / Y_scale_value;
  unsigned int rounded_divisor = nearestPowerOfTwo(Y_scale_value / (X_scale_value * W_scale_value));

  const auto* Xdata = X->template Data<int8_t>();
  const auto* Wdata = W->template Data<int8_t>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->template MutableData<int8_t>();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      TimePoint start_time;
      if (profiling_enabled) {
        start_time = profiler.StartTime();
      }

      if (col_buffer_data != nullptr) {
        if (kernel_rank == 2) {
          math::Im2col<int8_t, StorageOrder::NCHW>()(
              Xdata,
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
              col_buffer_data,
              X_zero_point_value);
        } else {
          math::Im2colNd<int8_t, StorageOrder::NCHW>()(
              Xdata,
              X->Shape().GetDims().data() + 1,
              col_buffer_shape.data(),
              C * input_image_size,
              col_buffer_size,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_rank),
              col_buffer_data,
              false,
              X_zero_point_value);
        }
        if (profiling_enabled) {
          profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                         Node().Name() + "_kernel_im2col_time",
                                         start_time,
                                         {{"op_name", KernelDef().OpName()},
                                          {"sub_action", "im2col"},
                                          {"provider", KernelDef().Provider()}});
          start_time = profiler.StartTime();
        }
      }


      std::unique_ptr<int32_t[]> broadcast_bias(nullptr);

      // Unlike the PyTorch implementation we cannot do bias at the end of the groups in a single multiplication
      // (Since the output from systolic is already quantized to int8 by that point)
      if (Bdata) {
        int dimI = static_cast<int>(M / conv_attrs_.group);
        int dimJ = static_cast<int>(output_image_size);
        std::unique_ptr<int[]> matrix_bias(new int[dimI * dimJ]);
        const int32_t* bias_data = Bdata + group_id * B_offset;
        for (int i = 0; i < dimI; i++) {
          std::fill(&matrix_bias.get()[i * dimJ], &matrix_bias.get()[i * dimJ + dimJ], bias_data[i]);
        }
        broadcast_bias = std::move(matrix_bias);
      }

      if (profiling_enabled) {
        profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                       Node().Name() + "_kernel_bias_splat_time",
                                       start_time,
                                       {{"op_name", KernelDef().OpName()},
                                        {"sub_action", "bias splat"},
                                        {"provider", KernelDef().Provider()}});
        start_time = profiler.StartTime();
      }

      SystolicMultiplyi8i8_i8(static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode(),
                              /*relu= */ fused_relu_,
                              static_cast<int>(M / conv_attrs_.group),
                              static_cast<int>(output_image_size),
                              static_cast<int>(kernel_dim),
                              Wdata + group_id * W_offset,
                              col_buffer_data == nullptr ? Xdata : col_buffer_data,
                              Ydata,
                              rounded_divisor, real_multiplier, broadcast_bias.get());

      if (profiling_enabled) {
        std::string dimension_string = std::to_string(static_cast<int>(M / conv_attrs_.group)) +
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
      }

      // GemmlowpDebug(W->template Data<int8_t>() + group_id * W_offset,
      //                   col_buffer_data,
      //                   Ydata + group_id * Y_offset,
      //                   *W_zero_point->template Data<int8_t>(),
      //                   *X_zero_point->template Data<int8_t>(),
      //                   *Y_zero_point->template Data<int8_t>(),
      //                   static_cast<int>(M / conv_attrs_.group),
      //                   static_cast<int>(output_image_size),
      //                   static_cast<int>(kernel_dim),
      //                   1,
      //                   rounded_divisor,
      //                   broadcast_bias.get());

      Xdata += X_offset;
      Ydata += Y_offset;
    }
  }

  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime