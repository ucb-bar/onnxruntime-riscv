// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/qlinearconv.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"

namespace onnxruntime {
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv<uint8_t, uint8_t, uint8_t>);

uint8_t saturate(int32_t num) {
    if (num < 0) return 0;
    if (num > 255) return 255;
    return num;
}

inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a,
                                                      std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 =
      static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

void mymatmul(int dimI, int dimJ, int dimK, uint8_t filteroffset, uint8_t input_offset, uint8_t resoffset, const uint8_t* in1, const uint8_t* in2, uint8_t* out, int integer_multiplier, int right_shift, const int32_t* bias = nullptr) {
    printf("Called my matmul\n");
    for (int i = 0; i < dimI; i++) {
        for (int j = 0; j < dimJ; j++) {
            int32_t res = 0;
            for (int k = 0; k < dimK; k++) {
                res += (in1[i * dimK + k] - filteroffset) * (in2[k * dimJ + j] - input_offset);
            }
            res = res + (bias != nullptr ? bias[i * dimJ + j] : 0);
            unsigned int rounding = (right_shift < 1) ? 0 : (1 << (right_shift - 1));
            int64_t interm = ((int64_t) SaturatingRoundingDoublingHighMul(integer_multiplier, res)) + rounding;
            out[i * dimJ + j] = saturate(resoffset + (interm >> right_shift));
        }
    }
}

void mymatmul(int dimI, int dimJ, int dimK, uint8_t filteroffset, uint8_t input_offset, uint8_t resoffset, const uint8_t* in1, const uint8_t* in2, uint8_t* out, float real_multiplier, const int32_t* bias = nullptr) {
    printf("Called my matmul\n");
    for (int i = 0; i < dimI; i++) {
        for (int j = 0; j < dimJ; j++) {
            int32_t res = 0;
            for (int k = 0; k < dimK; k++) {
                res += (in1[i * dimK + k] - filteroffset) * (in2[k * dimJ + j] - input_offset);
            }
            out[i * dimJ + j] = saturate(resoffset + real_multiplier * (res + (bias != nullptr ? bias[i * dimJ + j] : 0)));
        }
    }
}

template<>
Status QLinearConv<uint8_t, uint8_t, uint8_t>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(3);

  // validate offsets
  auto input_offset = context->Input<Tensor>(2);
  auto filter_offset = context->Input<Tensor>(5);
  auto result_offset = context->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(input_offset),
              "QLinearConv : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(filter_offset),
              "QLinearConv : filter zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(result_offset),
              "QLinearConv : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  auto input_scale = context->Input<Tensor>(1);
  auto filter_scale = context->Input<Tensor>(4);
  auto result_scale = context->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(input_scale),
              "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(filter_scale),
              "QLinearConv : filter scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(result_scale),
              "QLinearConv : result scale must be a scalar or 1D tensor of size 1");

  auto input_scale_data = *(input_scale->template Data<float>());
  auto filter_scale_data = *(filter_scale->template Data<float>());
  auto result_scale_data = *(result_scale->template Data<float>());

  const float real_multiplier = (input_scale_data * filter_scale_data) / result_scale_data;
  int32_t integer_multiplier;
  int right_shift;
  QuantizeMultiplier(real_multiplier, &integer_multiplier, &right_shift);

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* bias = nullptr;
  if (num_inputs == 9) {
    bias = context->Input<Tensor>(8);
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

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const auto* Xdata = X->template Data<uint8_t>();
  auto* Ydata = Y->template MutableData<uint8_t>();

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;
  const int bias_offset = static_cast<int>(M / conv_attrs_.group);

  auto col_data = alloc->Alloc(sizeof(uint8_t) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  auto* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

  TensorShape image_shape = X->Shape().Slice(1);
  std::vector<int64_t> col_buffer_shape{kernel_dim};
  col_buffer_shape.insert(col_buffer_shape.end(), output_shape.GetDims().begin(),
                          output_shape.GetDims().end());

  const size_t kernel_rank = kernel_shape.size();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (kernel_rank == 2) {
        math::Im2col<uint8_t, StorageOrder::NCHW>()(
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
            col_buffer_data,
            *input_offset->template Data<uint8_t>());
      } else {
        math::Im2colNd<uint8_t, StorageOrder::NCHW>()(
            Xdata + group_id * X_offset,
            image_shape.GetDims().data(),
            col_buffer_shape.data(),
            C * input_image_size,
            col_buffer_size,
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            static_cast<int>(kernel_shape.size()),
            col_buffer_data,
            false,
            *input_offset->template Data<uint8_t>());
      }

      std::unique_ptr<int32_t[]> broadcast_bias(nullptr);

      if (bias) {
        // for (int i = 0; i <  static_cast<int>(M / conv_attrs_.group); i++) {
        //   printf("%d ", (bias->template Data<int32_t>() + group_id * bias_offset)[i]);
        // }
        // printf("\n");
        int dimI = static_cast<int>(M / conv_attrs_.group);
        int dimJ = static_cast<int>(output_image_size);
        std::unique_ptr<int[]> matrix_bias(new int[dimI * dimJ]);
        const int32_t* bias_data = bias->template Data<int32_t>() + group_id * bias_offset;
        for (int i = 0; i < dimI; i++) {
          std::fill(&matrix_bias.get()[i * dimJ], &matrix_bias.get()[i * dimJ + dimJ], bias_data[i]);
        }
        broadcast_bias = std::move(matrix_bias);


        // Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_bias;
        // matrix_bias.resize(static_cast<int>(M / conv_attrs_.group), static_cast<int>(output_image_size));
        // ConstEigenVectorMap<int32_t> splatted(, static_cast<int>(M / conv_attrs_.group));
        // matrix_bias.colwise() = splatted;
        // broadcast_bias = matrix_bias.data();
      }

      printf("Called into gemmlowp conv\n");
      mymatmul(static_cast<int>(M / conv_attrs_.group),
                              static_cast<int>(output_image_size),
                              static_cast<int>(kernel_dim),
                              *filter_offset->template Data<uint8_t>(),
                              *input_offset->template Data<uint8_t>(),
                              *result_offset->template Data<uint8_t>(),
                              W->template Data<uint8_t>() + group_id * W_offset,
                              col_buffer_data,
                              Ydata + group_id * Y_offset,
                              real_multiplier, broadcast_bias.get());

      uint8_t mx = col_buffer_data[0];
      uint8_t mn = mx;
      for (int i = 0; i < static_cast<int>(kernel_dim) * static_cast<int>(output_image_size); i++) {
        mx = std::max(mx, col_buffer_data[i]);
        mn = std::min(mn, col_buffer_data[i]);
      }
      printf("max/min of input %d %d whose node is %s\n", mx, mn, Node().Name().c_str());

      mx = (Ydata + group_id * Y_offset)[0];
      mn = mx;
      for (int i = 0; i < static_cast<int>(M / conv_attrs_.group) * static_cast<int>(output_image_size); i++) {
        mx = std::max(mx, (Ydata + group_id * Y_offset)[i]);
        mn = std::min(mn, (Ydata + group_id * Y_offset)[i]);
      }

      printf("max/min of output %d %d whose node is %s\n", mx, mn, Node().Name().c_str());



      //if (Node().Name() == "vgg0_conv0_fwd_quant") {
        // GemmlowpDebug(W->template Data<uint8_t>() + group_id * W_offset,
        //                   col_buffer_data,
        //                   Ydata + group_id * Y_offset,
        //                   *filter_offset->template Data<uint8_t>(),
        //                   *input_offset->template Data<uint8_t>(),
        //                   *result_offset->template Data<uint8_t>(),
        //                   static_cast<int>(M / conv_attrs_.group),
        //                   static_cast<int>(output_image_size),
        //                   static_cast<int>(kernel_dim),
        //                   real_multiplier,
        //                   broadcast_bias.get());
      //}

      printf("\n----------------\n");
      // GemmlowpDebug(W->template Data<uint8_t>() + group_id * W_offset,
      //                         col_buffer_data,
      //                         Ydata + group_id * Y_offset,
      //                         *filter_offset->template Data<uint8_t>(),
      //                         *input_offset->template Data<uint8_t>(),
      //                         *result_offset->template Data<uint8_t>(),
      //                         static_cast<int>(M / conv_attrs_.group),
      //                         static_cast<int>(output_image_size),
      //                         static_cast<int>(kernel_dim),
      //                         integer_multiplier,
      //                         right_shift,
      //                         bias == nullptr ? nullptr : bias->template Data<int32_t>() + group_id * bias_offset);
    }

    Xdata += X_offset * conv_attrs_.group;
    Ydata += Y_offset * conv_attrs_.group;
  }

  return Status::OK();
}
}  // namespace onnxruntime
