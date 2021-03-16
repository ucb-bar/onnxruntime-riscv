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

template <typename T>
inline void transpose(const T* src, T* dst, size_t m, size_t n) {
  const size_t block = 32;
  for (size_t i = 0; i < m; i += block) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t b = 0; b < block && i + b < m; ++b) {
        dst[j * m + i + b] = src[(i + b) * n + j];
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
  transpose(in_vals, out_vals, O, H * W * I);
}

/**
 * Crash-course in NN Conv Backpropagation:
 * Let gradient with respect to weights be dW and grad wrt input be dX.
 * 
 * dW can be computed via im2col + gemm and dX via gemm + im2col
 * 
 * Some tidbits to help understand how the dW computation works:
 * https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa
 * In simple-case (non-strided, non-padded we have)
 * dW = IHWO_to_HWIO(conv2d(input=NHWC_TO_CHWN(x), filter=NHWC_TO_HWNC(dilated(dY)))
 * (which can be expressed only in terms of 2D transpositions by summing across batch in software)
 *   dW = IHWO_to_HWIO(sum([conv2d(input=1HWC_TO_CHW1(x_n), filter=1HWC_TO_HW1C(dilated(dY))) for n in N]))
 * Transposing the im2col effectively gives as the columns the elements that a dilated kernel would have acted upon
 * (work this out for the simple examples shown in the unit test)
 * 
 * Then when dealing with multiple channels and multiple batches, we swap channel and batch dimensions
 * (doing each channel independently and accumulating the gradient over the batches)
 * Thus we arrive at the neat formula `dW = transpose(im2col(X) @ transpose(dY))`
 * For an illustration see https://hackmd.io/@bouteille/B1Cmns09I#%E2%97%8B-Kernel-gradient-Intuition
 * 
 * The operation that computes `dX` is known in the literature by names such as ConvTranspose or fractionally strided conv
 * More info about the operator: https://d2l.ai/chapter_computer-vision/transposed-conv.html
 * You can see why it's called "conv transpose" here: https://stackoverflow.com/a/64732177/2612743
 * 
 * The generalization of the above results in the gemm + col2im method of computing
 * (see https://stackoverflow.com/a/64457058 for why the col2im is needed)
 * https://hackmd.io/@bouteille/B1Cmns09I#%E2%8B%86-Layer-gradient-Intuition has a good graphical explanation
 * (but ignore the formula presented, since it should be convtranspose not conv)
 * 
 * https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
 * Is an excellent article describing everything you need to know about ConvTranspose.
 * Combining the above and these two:
 * https://www.adityaagrawal.net/blog/deep_learning/bprop_strided_conv
 * https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
 * 
 * You should have everything you need to understand how to express a ConvTranspose in terms of a Conv
 * See https://gist.github.com/pranav-prakash/08f66af9ac62ab408261f5a479ceae13
 * for the equivalency. But the gist is that you dilate the input, rotate the kernel 180deg (reverse both H and W dims),
 * then perform a conv with FULL padding
 */

template <typename T>
Status ConvGrad_nhwc<T>::Compute(OpKernelContext* context) const {
  printf("IN SYSTOLIC CONVGRAD NHWC\n");
  char acc_mode = static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode();
  profiling::Profiler& profiler = static_cast<OpKernelContextInternal*>(context)->GetProfiler();
  bool profiling_enabled = profiler.IsEnabled();

  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);

  printf("dY");
  PrintMinMax(dY->Shape().Size(), dY->template Data<T>());
  printf("X");
  PrintMinMax(X->Shape().Size(), X->template Data<T>());

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
  TimePoint dW_start;
  if (profiling_enabled) {
    dW_start = profiler.StartTime();
  }

  // We loop over all the images, and accumulate the gradient for each.
  // Note how in the Gemm we add into the existing.
  for (int image_id = 0; image_id < N; ++image_id) {
    TimePoint start_time;
    if (profiling_enabled) {
      start_time = profiler.StartTime();
    }

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

    if (profiling_enabled) {
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     Node().Name() + "_dw_im2col",
                                     start_time,
                                     {{"op_name", KernelDef().OpName()},
                                      {"sub_action", "dw_im2col"},
                                      {"provider", KernelDef().Provider()}});
      start_time = profiler.StartTime();
    }

    // Here the "weight" of this convolution, dY is NHWC. We accumulate across the batches in the outer loop.
    // In this inner loop the channels are used as output channels. Note that we transpose yData so HWxC -> CxHW
    // We can then multiply (C x HW) * (im2col of X) to get an OHWI output
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      SystolicGemm(acc_mode, /*transA= */ true, /*transB= */ false,
                   static_cast<int>(M / conv_attrs_.group),  // Do one matrix-vector product per output channel
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

    if (profiling_enabled) {
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     Node().Name() + "_dw_gemm",
                                     start_time,
                                     {{"op_name", KernelDef().OpName()},
                                      {"sub_action", "dw_gemm"},
                                      {"provider", KernelDef().Provider()}});
      start_time = profiler.StartTime();
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

  if (profiling_enabled) {
    profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                   Node().Name() + "_dW_and_dB",
                                   dW_start,
                                   {{"op_name", KernelDef().OpName()},
                                    {"sub_action", "_dW_and_dB"},
                                    {"provider", KernelDef().Provider()}});
  }

  //printf("dW_ohwi");
  //PrintMinMax(dW->Shape().Size(), ohwi_dW_data);
  // At this point ohwi_dW_data is formatted as [output_channels (derived from M), h, w, input channels]
  OHWItoHWIO(ohwi_dW_data, dWdata, dW->Shape());
  // printf("\n");
  //printf("dW finished\n");
  // DumpTensor<float>(dW);

  // Now we proceed to calculate dX

  TimePoint dX_start;
  if (profiling_enabled) {
    dX_start = profiler.StartTime();
  }

  Tensor* dX = context->Output(0, X->Shape());
  if (dX) {
    T* dXdata = dX->template MutableData<T>();
    dYdata = dY->template Data<T>();
    for (int image_id = 0; image_id < N; ++image_id) {
      TimePoint start_time;
      if (profiling_enabled) {
        start_time = profiler.StartTime();
      }
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
                     M / conv_attrs_.group,
                     0,
                     col_buffer_data + group_id * kernel_dim,
                     conv_attrs_.group * kernel_dim);
        GemmlowpDebug(
            /*transA= */ false,
            /*transB= */ true,
            output_image_size,
            kernel_dim,
            M / conv_attrs_.group,
            dYdata + Y_offset * image_id + group_id * (M / conv_attrs_.group),
            M,
            Wdata + group_id * (M / conv_attrs_.group) * kernel_dim,
            M / conv_attrs_.group,
            col_buffer_data + group_id * kernel_dim,
            conv_attrs_.group * kernel_dim);
      }

      if (profiling_enabled) {
        profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                       Node().Name() + "_dX_gemm",
                                       start_time,
                                       {{"op_name", KernelDef().OpName()},
                                        {"sub_action", "_dX_gemm"},
                                        {"provider", KernelDef().Provider()}});
        start_time = profiler.StartTime();
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

      if (profiling_enabled) {
        profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                       Node().Name() + "_dX_col2im",
                                       start_time,
                                       {{"op_name", KernelDef().OpName()},
                                        {"sub_action", "_dX_col2im"},
                                        {"provider", KernelDef().Provider()}});
        start_time = profiler.StartTime();
      }

      dXdata += X_offset;
    }
  }

  if (profiling_enabled) {
    profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                   Node().Name() + "_dX",
                                   dX_start,
                                   {{"op_name", KernelDef().OpName()},
                                    {"sub_action", "_dX"},
                                    {"provider", KernelDef().Provider()}});
  }

  // // printf("\n");
  // printf("dX finished\n");
  // // DumpTensor<float>(dX);

  printf("dX\n");
  //DumpTensor<float>(dX);
  PrintMinMax<float>(dX);
  printf("dW\n");
  // DumpTensor<float>(dW);
  PrintMinMax<float>(dW);
  printf("dB\n");
  // DumpTensor<float>(dB);
  PrintMinMax<float>(dB);

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