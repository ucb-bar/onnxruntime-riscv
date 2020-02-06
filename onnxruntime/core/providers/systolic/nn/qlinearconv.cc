// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinearconv.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"

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
    QLinearConv<int8_t, int8_t, int8_t>);


int nearestPowerOfTwo(int n)
{
    int v = n; 

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++; // next power of 2

    int x = v >> 1; // previous power of 2

    return (v - n) > (n - x) ? x : v;
}

#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ({((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)));})

inline int8_t saturate(int32_t num, int shift) {
    num = ROUNDING_RIGHT_SHIFT(num, shift);
    // const int divisor = 1 << shift;
    // int32_t abs = num > 0 ? num : -num;
    // int32_t shifted = (abs + (divisor/2)) / divisor;
    // if (num < 0)
    //     num = -shifted;
    // else
    //     num = shifted;
    // Clip result
    return num > SCHAR_MAX ? SCHAR_MAX : (num < SCHAR_MIN ? SCHAR_MIN : num);
}

inline void mymatmul(int dimI, int dimJ, int dimK, const int8_t* in1, const int8_t* in2, int8_t* out, float real_multiplier, const int32_t* bias = nullptr) {
    printf("Called my matmul\n");
    for (int i = 0; i < dimI; i++) {
        for (int j = 0; j < dimJ; j++) {
            int32_t res = 0;
            for (int k = 0; k < dimK; k++) {
                res += in1[i * dimK + k] * in2[k * dimJ + j];
            }
            out[i * dimJ + j] = saturate(real_multiplier * (res + (bias != nullptr ? bias[i * dimJ + j] : 0)), 0);
        }
    }
}

template <>
Status QLinearConv<int8_t, int8_t, int8_t>::Compute(OpKernelContext* context) const {
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

  auto input_offset_data = *(input_offset->template Data<int8_t>());
  auto filter_offset_data = *(filter_offset->template Data<int8_t>());
  auto result_offset_data = *(result_offset->template Data<int8_t>());
  ORT_ENFORCE(input_offset_data == 0, "Systolic can only handle zero offset for input");
  ORT_ENFORCE(filter_offset_data == 0, "Systolic can only handle zero offset for filter");
  ORT_ENFORCE(result_offset_data == 0, "Systolic can only handle zero offset for result");

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
  //ORT_ENFORCE(input_scale_data == 1, "Systolic can only handle scale of 1 for input");
  auto filter_scale_data = *(filter_scale->template Data<float>());
  //ORT_ENFORCE(filter_scale_data == 1, "Systolic can only handle scale of 1 for filter");
  auto result_scale_data = *(result_scale->template Data<float>());


  ORT_ENFORCE(result_scale_data != 0, "result_scale_data cannot be 0");
  ORT_ENFORCE(filter_scale_data != 0, "filter_scale_data cannot be 0");
  ORT_ENFORCE(input_scale_data != 0, "input_scale_data cannot be 0");

  const float real_multiplier = (input_scale_data * filter_scale_data) / result_scale_data;
  // int32_t integer_multiplier;
  // int right_shift;
  //QuantizeMultiplier(real_multiplier, &integer_multiplier, &right_shift);
  
  //unsigned int right_shift = nearestPowerOfTwo(result_scale_data / (input_scale_data * filter_scale_data));

  // ORT_ENFORCE(result_scale_data - (int)result_scale_data <= 1E-5, "Systolic can only handle integer divisors for result scale");
  // int result_scale_data_rounded = (int)result_scale_data;
  // ORT_ENFORCE(result_scale_data_rounded && !(result_scale_data_rounded & (result_scale_data_rounded - 1)), "Systolic can only handle power of 2 divisor for result scale");

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* bias = nullptr;
  if (num_inputs == 9) {
    bias = context->Input<Tensor>(8);
  }
  //ORT_ENFORCE(bias == nullptr, "Systolic cannot handle bias in conv");

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

  const auto* Xdata = X->template Data<int8_t>();
  auto* Ydata = Y->template MutableData<int8_t>();

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;
  const int bias_offset = static_cast<int>(M / conv_attrs_.group);

  auto col_data = alloc->Alloc(sizeof(int8_t) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  auto* col_buffer_data = static_cast<int8_t*>(col_buffer.get());

  TensorShape image_shape = X->Shape().Slice(1);
  std::vector<int64_t> col_buffer_shape{kernel_dim};
  col_buffer_shape.insert(col_buffer_shape.end(), output_shape.GetDims().begin(),
                          output_shape.GetDims().end());

  const size_t kernel_rank = kernel_shape.size();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (kernel_rank == 2) {
        math::Im2col<int8_t, StorageOrder::NCHW>()(
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
            *input_offset->template Data<int8_t>());
      } else {
        math::Im2colNd<int8_t, StorageOrder::NCHW>()(
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
            *input_offset->template Data<int8_t>());
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


      if (false && static_cast<int>(M / conv_attrs_.group) % 16 == 0 && static_cast<int>(output_image_size) % 16 == 0 && static_cast<int>(kernel_dim) % 16 == 0) {
        printf("Using systolic accelerated matmul for %s with input_scale %f, filter_scale %f, result_scale %f\n", Node().Name().c_str(), input_scale_data, filter_scale_data, result_scale_data);
        unsigned int right_shift = nearestPowerOfTwo(result_scale_data / (input_scale_data * filter_scale_data));
        printf("Right shift: %d", right_shift);
        SystolicMultiplyi8i8_i8(static_cast<int>(M / conv_attrs_.group),
                        static_cast<int>(output_image_size),
                        static_cast<int>(kernel_dim),
                        W->template Data<int8_t>() + group_id * W_offset,
                        col_buffer_data,
                        Ydata + group_id * Y_offset,
                        right_shift, broadcast_bias.get());

      } else {
        printf("Using cpu accelerated matmul for %s with input_scale %f, filter_scale %f, result_scale %f\n", Node().Name().c_str(), input_scale_data, filter_scale_data, result_scale_data);
        mymatmul(static_cast<int>(M / conv_attrs_.group),
                              static_cast<int>(output_image_size),
                              static_cast<int>(kernel_dim),
                              W->template Data<int8_t>() + group_id * W_offset,
                              col_buffer_data,
                              Ydata + group_id * Y_offset,
                              real_multiplier, broadcast_bias.get());

      }

      int8_t mx = col_buffer_data[0];
      int8_t mn = mx;
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



      // if (Node().Name() == "vgg0_conv0_fwd_quant") {
      //   GemmlowpDebug(W->template Data<int8_t>() + group_id * W_offset,
      //                     col_buffer_data,
      //                     Ydata + group_id * Y_offset,
      //                     *filter_offset->template Data<int8_t>(),
      //                     *input_offset->template Data<int8_t>(),
      //                     *result_offset->template Data<int8_t>(),
      //                     static_cast<int>(M / conv_attrs_.group),
      //                     static_cast<int>(output_image_size),
      //                     static_cast<int>(kernel_dim),
      //                     real_multiplier,
      //                     broadcast_bias.get());
      // }


    }

    Xdata += X_offset * conv_attrs_.group;
    Ydata += Y_offset * conv_attrs_.group;
  }

  return Status::OK();
}

} // namespace systolic
}  // namespace onnxruntime
