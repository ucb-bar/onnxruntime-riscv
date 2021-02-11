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

#include "conv_grad.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"
#include "core/providers/systolic/nn/pool_attributes.h"
#include "core/providers/systolic/nn/conv_pool_helper.h"

#ifdef SYSTOLIC_FP32

namespace onnxruntime {
namespace systolic {

template<typename T>
inline void transpose(const T *src, T *dst, size_t m, size_t n) {
  const size_t block = 32;
  for (size_t i = 0; i < m; i += block) {
      for(size_t j = 0; j < n; ++j) {
          for(size_t b = 0; b < block && i + b < m; ++b) {
              dst[j*m + i + b] = src[(i + b)*n + j];
          }
      }
  }
}

/**
 * After the im2col and gemm for dW, we get the result in NHWC format.
 * NHWC here is equivalent to OHWI format. We convert this to HWIO (HWCN)
 * 
 * Converts [batch, height, widht, channels]
 * to [filter_height, filter_width, in_channels, out_channels]
 * i.e. batch -> out_channels, channels -> in_channels
 */
template <typename T>
inline void OHWItoHWIO(const T* in_vals, T* out_vals, const TensorShape& output_shape) {
  int H = output_shape[0];
  int W = output_shape[1];
  int I = output_shape[2];
  int O = output_shape[3];
  // Note that because we are doing OHWI -> HWIO where we just move the single axis O inwards,
  // we can treat this as a O x HWI matrix and just do a 2D transpose.
  transpose(in_vals, out_vals, O, H*W*I);
}

template <typename T>
Status ConvGrad_nhwc<T>::Compute(OpKernelContext* context) const {
  printf("IN SYSTOLIC CONVGRAD NHWC\n");
  char acc_mode = static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode();

  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[3];
  const int64_t M = W->Shape()[3];

  ORT_RETURN_IF_ERROR(ValidateConvInputShapeNHWC(X, W, conv_attrs_.group));

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

  Tensor* dW = context->Output(1, W->Shape());
  T* dWdata = dW->template MutableData<T>();

  TensorShape input_shape = X->Shape().Slice(1, 3);
  TensorShape output_shape = dY->Shape().Slice(1, 3);

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C * input_image_size;
  const int64_t Y_offset = (dY->Shape().Size() / dY->Shape()[0]);
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = C * output_image_size * kernel_size;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  BufferUniquePtr col_buffer(alloc->Alloc(sizeof(T) * col_buffer_size), BufferDeleter(alloc));
  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  const T* Xdata = X->template Data<T>();
  const T* Wdata = W->template Data<T>();
  const T* dYdata = dY->template Data<T>();

  // Note that we allocate a temporary buffer to accumulate into, after which we convert back to NHWC
  BufferUniquePtr ohwi_dW(alloc->Alloc(sizeof(T) * dW->Shape().Size()), BufferDeleter(alloc));
  T* ohwi_dW_data = static_cast<T*>(ohwi_dW.get());

  math::Set<T, CPUMathUtil>(dW->Shape().Size(), 0, ohwi_dW_data, &CPUMathUtil::Instance());
  BufferUniquePtr bias_multiplier(alloc->Alloc(sizeof(T) * output_image_size), BufferDeleter(alloc));
  T* bias_multiplier_data = nullptr;
  Tensor* dB = context->Output(2, TensorShape({M}));
  T* dBdata = nullptr;
  if (dB) {
    dBdata = dB->template MutableData<T>();
    math::Set<T, CPUMathUtil>(dB->Shape().Size(), static_cast<T>(0), dBdata, &CPUMathUtil::Instance());

    bias_multiplier_data = static_cast<T*>(bias_multiplier.get());
    math::Set<T, CPUMathUtil>(output_image_size,
                              static_cast<T>(1),
                              bias_multiplier_data,
                              &CPUMathUtil::Instance());
  }

  // We first calculate dW

  // We loop over all the images, and accumulate the gradient for each.
  // Note how in the Gemm we add into the existing.
  for (int image_id = 0; image_id < N; ++image_id) {
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
        (float)0.0);

    // Here the "weight" of this convolution, dY is NHWC. We accumulate across the batches in the outer loop.
    // In this inner loop the channels are used as output channels. Note that we transpose yData so HWxC -> CxHW
    // We can then multiply (C x HW) * (im2col of X) to get an OHWI output
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      SystolicGemm(acc_mode, /*transA= */ true, /*transB= */ false,
                   static_cast<int>(M / conv_attrs_.group), // Do one matrix-vector product per output channel
                   static_cast<int>(kernel_dim),
                   static_cast<int>(output_image_size),
                   1,
                   dYdata + Y_offset * image_id + group_id * (M / conv_attrs_.group),
                   M,
                   col_buffer_data + group_id * kernel_dim,
                   conv_attrs_.group * kernel_dim,
                   1,
                   ohwi_dW_data + group_id * (M / conv_attrs_.group) * kernel_dim,
                   kernel_dim);
      // GemmlowpDebug(/*transA= */ true, /*transB= */ false,
      //               static_cast<int>(M / conv_attrs_.group),
      //               static_cast<int>(kernel_dim),
      //               static_cast<int>(output_image_size),
      //               dYdata + Y_offset * image_id + group_id * (M / conv_attrs_.group),
      //               M,
      //               col_buffer_data + group_id * kernel_dim,
      //               conv_attrs_.group * kernel_dim,
      //               ohwi_dW_data + group_id * (M / conv_attrs_.group) * kernel_dim,
      //               kernel_dim);
    }
    if (dB) {
      // Gradient with respect to bias can be computed independent from group.
      SystolicGemm(acc_mode,
                   /*transA=*/true, /*transB= */ false,
                   static_cast<int>(M),
                   1,
                   static_cast<int>(output_image_size),
                   1,
                   dYdata + Y_offset * image_id,
                   bias_multiplier_data,
                   1,
                   dBdata);
    }
    Xdata += X_offset;
  }

  printf("dW_ohwi");
  PrintMinMax( dW->Shape().Size(), ohwi_dW_data);
  // At this point ohwi_dW_data is formatted as [output_channels (derived from M), h, w, input channels]
  OHWItoHWIO(ohwi_dW_data, dWdata, dW->Shape());
  // printf("\n");
  printf("dW finished\n");
  // DumpTensor<float>(dW);

  // Now we proceed to calculate dX

  Tensor* dX = context->Output(0, X->Shape());
  if (dX) {
    T* dXdata = dX->template MutableData<T>();
    dYdata = dY->template Data<T>();
    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
        SystolicGemm(acc_mode,
                     /*transA= */ false,
                     /*transB= */ true,
                     output_image_size,
                     kernel_dim,
                     M / conv_attrs_.group,
                     1,
                     dYdata + Y_offset * image_id + group_id * (M / conv_attrs_.group),
                     M,
                     Wdata + group_id * (M / conv_attrs_.group) * kernel_dim,
                     kernel_dim,
                     0,
                     col_buffer_data + group_id * kernel_dim,
                     conv_attrs_.group * kernel_dim);

        // GemmlowpDebug(
        //     output_image_size,
        //     kernel_dim,
        //     M / conv_attrs_.group,
        //     dYdata + Y_offset * image_id + group_id * (M / conv_attrs_.group),
        //     M,
        //     Wdata + group_id * (M / conv_attrs_.group) * kernel_dim,
        //     kernel_dim,
        //     col_buffer_data + group_id * kernel_dim,
        //     conv_attrs_.group * kernel_dim, 1, nullptr, 0);
      }

      Col2Im_NHWC(
          col_buffer_data,
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
          dXdata,
          conv_attrs_.group);

      dXdata += X_offset;
    }
  }

  // // printf("\n");
  // printf("dX finished\n");
  // // DumpTensor<float>(dX);


  // printf("dX\n");
  // DumpTensor<float>(dX);
  // //PrintMinMax<float>(dX);
  // printf("dW\n");
  // DumpTensor<float>(dW);
  // //PrintMinMax<float>(dW);
  // printf("dB\n");
  // DumpTensor<float>(dB);
  // //PrintMinMax<float>(dB);

  return Status::OK();
}  // namespace systolic

template <typename T>
Status ConvGrad<T>::Compute(OpKernelContext* context) const {
  printf("IN SYSTOLIC CONVGRAD\n");
  char acc_mode = static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode();

  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];

  // TODO: validataion might not be needed, since it's already done once in the fw pass
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  // Copied from conv_impl.h, maybe refactor
  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  bool Is2DKernel = kernel_shape.size() == 2;
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

  Tensor* dW = context->Output(1, W->Shape());
  T* dWdata = dW->template MutableData<T>();

  TensorShape input_shape = X->Shape().Slice(2);
  TensorShape output_shape = dY->Shape().Slice(2);

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = dY->Shape().Size() / dY->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr col_buffer(alloc->Alloc(sizeof(T) * col_buffer_size), BufferDeleter(alloc));
  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  const T* Xdata = X->template Data<T>();
  const T* Wdata = W->template Data<T>();
  const T* dYdata = dY->template Data<T>();

  // Pre-setting the gradients to zero.
  math::Set<T, CPUMathUtil>(dW->Shape().Size(), 0, dWdata, &CPUMathUtil::Instance());

  BufferUniquePtr bias_multiplier(alloc->Alloc(sizeof(T) * output_image_size), BufferDeleter(alloc));
  T* bias_multiplier_data = nullptr;
  Tensor* dB = context->Output(2, TensorShape({M}));
  T* dBdata = nullptr;
  if (dB) {
    dBdata = dB->template MutableData<T>();
    math::Set<T, CPUMathUtil>(dB->Shape().Size(), static_cast<T>(0), dBdata, &CPUMathUtil::Instance());

    bias_multiplier_data = static_cast<T*>(bias_multiplier.get());
    math::Set<T, CPUMathUtil>(output_image_size,
                              static_cast<T>(1),
                              bias_multiplier_data,
                              &CPUMathUtil::Instance());
  }

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (Is2DKernel) {
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
      // Gradient with respect to W, filter.

      SystolicGemm(acc_mode, /*transA=*/false, /*transB= */ true,
                   M / conv_attrs_.group,
                   kernel_dim,
                   output_image_size,
                   1,
                   dYdata + group_id * Y_offset,
                   col_buffer_data,
                   1,
                   dWdata + group_id * W_offset);

      //GemmlowpDebug(M / conv_attrs_.group, kernel_dim, output_image_size, dYdata + group_id * Y_offset, col_buffer_data,  dWdata + group_id * W_offset);
    }
    if (dB) {
      printf("dB Provided\n");
      // Gradient with respect to bias can be computed independent from group.
      SystolicGemm(acc_mode,
                   /*transA=*/false, /*transB= */ false,
                   static_cast<int>(M),
                   1,
                   static_cast<int>(output_image_size),
                   1,
                   dYdata,
                   bias_multiplier_data,
                   1,
                   dBdata);
    }
    Xdata += X_offset * conv_attrs_.group;
    dYdata += Y_offset * conv_attrs_.group;
  }

  Tensor* dX = context->Output(0, X->Shape());
  if (dX) {
    T* dXdata = dX->template MutableData<T>();
    dYdata = dY->template Data<T>();
    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
        // Compute gradient into col_buffer.
        SystolicGemm(acc_mode, /*transA= */ true, /*transB= */ false,
                     kernel_dim,
                     output_image_size,
                     M / conv_attrs_.group,
                     1,
                     Wdata + group_id * W_offset,
                     dYdata,
                     0,
                     col_buffer_data);

        // GemmlowpDebug(
        //     kernel_dim,
        //     output_image_size,
        //     M / conv_attrs_.group,
        //     Wdata + group_id * W_offset,
        //     dYdata,
        //     col_buffer_data);

        if (Is2DKernel) {
          math::Col2im<T, CPUMathUtil, StorageOrder::NCHW>(
              col_buffer_data,
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
              dXdata,
              &CPUMathUtil::Instance());
        } else {
          math::Col2imNd<T, CPUMathUtil, StorageOrder::NCHW>(
              col_buffer_data,
              input_shape.GetDims().data(),
              output_shape.GetDims().data(),
              kernel_dim,
              C * input_image_size,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_shape.size()),
              dXdata,
              &CPUMathUtil::Instance());
        }
        dXdata += X_offset;
        dYdata += Y_offset;
      }
    }
  }
  printf("dX\n");
  //DumpTensor<float>(dX);
  PrintMinMax<float>(dX);
  printf("dW\n");
  PrintMinMax<float>(dW);
  printf("dB\n");
  PrintMinMax<float>(dB);
  // DumpTensor<float>(dW);
  // printf("dB\n");
  // DumpTensor<float>(dB);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    ConvGrad,
    kOnnxDomain,
    9,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvGrad<float>);

ONNX_OPERATOR_KERNEL_EX(
    ConvGrad_nhwc,
    kOnnxDomain,
    9,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvGrad_nhwc<float>);

}  // namespace systolic
}  // namespace onnxruntime

#endif