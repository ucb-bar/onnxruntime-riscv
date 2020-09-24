// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_quant.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"

using onnxruntime::concurrency::ThreadPool;

// #define PRINT_QUANTIZATION_SCALES

#if defined(SYSTOLIC_INT8) && !defined(DISABLE_CONTRIB_OPS)

namespace onnxruntime {
namespace systolic {

#ifdef PRINT_QUANTIZATION_SCALES

void PrintQuantizationScale(const float* arr, size_t length, int type, const char* node_name) {
    auto mn_mx = std::minmax_element(arr, arr + length);
    printf("QUANT_OUT %s %d %f %f\n", node_name, type, *mn_mx.first, *mn_mx.second);
    return;
}

float MultiplyForQuantScalePrint(int dimI, int dimJ, int dimK,
                             const int8_t* in1, int strideIn1,
                             const int8_t* in2, int strideIn2,
                             int8_t* out, int strideOut,
                             float real_multiplier) {

    std::unique_ptr<float[]> temp = std::make_unique<float[]>(dimI * dimJ);
    for (int i = 0; i < dimI; i++) {
      for (int j = 0; j < dimJ; j++) {
        int32_t result = 0;

        for (int k = 0; k < dimK; k++) {
            result += (*(in1 + i * strideIn1 + k)) * (*(in2 + k * strideIn2 + j));
        }
        *(temp.get() + i * dimJ + j) = result * real_multiplier;
      }
    }
    auto mn_mx = std::minmax_element(temp.get(), temp.get() + dimI * dimJ);

    float scale = std::max(abs(*mn_mx.first), abs(*mn_mx.second)) / 127.0;

    for (int i = 0; i < dimI; i++) {
      for (int j = 0; j < dimJ; j++) {
        int x = (int) std::nearbyintf(*(temp.get() + i *dimJ + j) / scale );
        *(out + i * strideOut + j)  = x > 127 ? 127 : (x < -128 ? -128 : x);
      }
    }
    return scale;
}

#endif

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QAttention,
    kMSDomain,
    1,
    float,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QAttention<float>);

template <typename T>
Status QAttention<T>::Compute(OpKernelContext* context) const {
  // Input and output shapes:
  //   Input  0 - input             : (batch_size, sequence_length, hidden_size)
  //   Input  1 - weights           : (hidden_size, 3 * hidden_size)
  //   Input  2 - bias              : (3 * hidden_size)
  //   Input  3 - input_scale       : scalar
  //   Input  4 - weight_scale      : scalar
  //   Input  5 - mask_index        : nullptr, (batch_size), (2 * batch_size), (batch_size, 1), (1, 1) or (batch_size, past_sequence_length + sequence_length)
  //   Input  6 - input_zero_point  : scalar
  //   Input  7 - weight_zero_point : scalar
  //   Input  8 - past              : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   Output 0                     : (batch_size, sequence_length, hidden_size)
  //   Output 1 - present           : (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)
  //   ORT_RETURN_IF_ERROR(CheckInputs(context));
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = packed_weights_ ? nullptr : context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* input_scale_tensor = context->Input<Tensor>(3);
  const Tensor* weight_scale_tensor = context->Input<Tensor>(4);
  const Tensor* mask_index = context->Input<Tensor>(5);
  const Tensor* i_zp_tensor = context->Input<Tensor>(6);
  const Tensor* w_zp_tensor = context->Input<Tensor>(7);
  const Tensor* past_tensor = context->Input<Tensor>(8);

  const Tensor* q_scale_tensor = context->Input<Tensor>(9);
  const Tensor* k_scale_tensor = context->Input<Tensor>(10);
  const Tensor* v_scale_tensor = context->Input<Tensor>(11);

  ORT_RETURN_IF_ERROR(AttentionBase::CheckInputs(input->Shape(),
                                                 packed_weights_ ? weight_shape_ : weights->Shape(),
                                                 bias->Shape(),
                                                 mask_index,
                                                 past_tensor));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  T input_scale = *(input_scale_tensor->template Data<T>());

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(weight_scale_tensor),
                    "weight must be a scalar or 1D tensor of size 1");
  T weight_scale = *(weight_scale_tensor->template Data<T>());


  bool has_qkv_scale = false;
  T q_scale = 1;
  T k_scale = 1;
  T v_scale = 1;
  if (q_scale_tensor != nullptr) {
    ORT_RETURN_IF_NOT(k_scale_tensor != nullptr && v_scale_tensor != nullptr,
                      "Must provide all are none of q/k/v scale");
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(q_scale_tensor),
                      "q must be a scalar or 1D tensor of size 1");
    has_qkv_scale = true;
    q_scale = *(q_scale_tensor->template Data<T>());

      ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(k_scale_tensor),
                      "q must be a scalar or 1D tensor of size 1");
    k_scale = *(k_scale_tensor->template Data<T>());

    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(v_scale_tensor),
                      "q must be a scalar or 1D tensor of size 1");
    v_scale = *(v_scale_tensor->template Data<T>());
  }


#ifndef PRINT_QUANTIZATION_SCALES
  ORT_RETURN_IF_NOT(has_qkv_scale,
                      "Must provide qkv scale unless quantization #ifdef set");
#else
  ORT_UNUSED_PARAMETER(has_qkv_scale);
#endif

  //T dequant_scale = input_scale * weight_scale;

  int8_t input_zero_point = 0;
  if (i_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(i_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    input_zero_point = *i_zp_tensor->template Data<int8_t>();
    ORT_ENFORCE(input_zero_point == 0, "Systolic can only handle zero offset for input");
  }

  int8_t weight_zero_point = 0;
  if (w_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(w_zp_tensor),
                      "weight zero point must be a scalar or 1D tensor of size 1.");
    weight_zero_point = *static_cast<const int8_t*>(w_zp_tensor->DataRaw());
    ORT_ENFORCE(weight_zero_point == 0, "Systolic can only handle zero offset for weight");
  }

  const auto& shape = input->Shape();
  const int batch_size = static_cast<int>(shape[0]);
  const int sequence_length = static_cast<int>(shape[1]);
  const int hidden_size = static_cast<int>(shape[2]);
  const int head_size = hidden_size / num_heads_;

  Tensor* output = context->Output(0, shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  constexpr size_t element_size = sizeof(T);

  auto* tp = context->GetOperatorThreadPool();
  // STEP.1: gemm_data(BS, 3NH) = Scale(input(BS, NH) x weights(NH, 3NH)) + bias(3NH)
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + batch_size * sequence_length * hidden_size;
  auto V = K + batch_size * sequence_length * hidden_size;
  T* QKV[3] = {Q, K, V};
  float output_scale[3] = {q_scale, k_scale, v_scale};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->template Data<int8_t>();
    const auto* bias_data = bias->template Data<T>();

    const auto* weights_data = packed_weights_ ? nullptr : static_cast<const int8_t*>(weights->DataRaw());

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        int input_offset = batch_index * sequence_length * hidden_size;
        int weights_offset = qkv_index * hidden_size + head_index * head_size;
        float* qkv_dest = QKV[qkv_index];
        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

        //                   original           transposed            iteration
        // A: input          (BxSxNxH)          (B.)S x NH            S x NH
        // B: weights        (NxHx3xNxH)        NH  x (3.N.)H         NH x H
        // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H

        auto c_int8 = std::make_unique<int8_t[]>(sequence_length * head_size);

#ifdef PRINT_QUANTIZATION_SCALES

    float returned_scale = MultiplyForQuantScalePrint(
                    sequence_length,                // M      = S
                    head_size,                      // N      = H
                    hidden_size,                    // K      = NH
                    input_data + input_offset,      // A
                    hidden_size,                    // lda    = NH
                    weights_data + weights_offset,  // B
                    3 * hidden_size,                // ldb    = 3NH
                    c_int8.get(),                   // C
                    head_size,                      // ldc
                    weight_scale * input_scale     // real multiplier
  );
  output_scale[qkv_index] = returned_scale;
#else

        SystolicMultiply(static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode(),
                         /*relu= */ false,
                         sequence_length,                // M      = S
                         head_size,                      // N      = H
                         hidden_size,                    // K      = NH
                         input_data + input_offset,      // A
                         hidden_size,                    // lda    = NH
                         weights_data + weights_offset,  // B
                         3 * hidden_size,                // ldb    = 3NH
                         c_int8.get(),                   // C
                         head_size,                      // ldc
                         weight_scale * input_scale / output_scale[qkv_index],  // real multiplier
                         nullptr,                        // bias
                         0,                              // strideBias
                         false                           // repeating bias
        );
#endif

        {
          int M = sequence_length;
          int N = head_size;
          int ldc = head_size;
          int8_t *raw_int8_data = c_int8.get();
          float* result_data = qkv_dest + qkv_offset;
          const float* bias = bias_data + weights_offset;
          for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                result_data[n] = static_cast<float>(raw_int8_data[n]) * output_scale[qkv_index] + bias[n];
            }
            result_data += ldc;
            raw_int8_data += head_size;
          }
        }
      }
    });
  }

#ifdef PRINT_QUANTIZATION_SCALES
      std::string name = this->Info().node().Name();
      PrintQuantizationScale(Q, batch_size * sequence_length * hidden_size, 0, name.c_str());
      PrintQuantizationScale(K, batch_size * sequence_length * hidden_size, 1, name.c_str());
      PrintQuantizationScale(V, batch_size * sequence_length * hidden_size, 2, name.c_str());
#endif

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q, K, V, mask_index, past_tensor, output,
                        batch_size, sequence_length,
                        head_size, hidden_size, context);
}

}  // namespace systolic
}  // namespace onnxruntime

#endif