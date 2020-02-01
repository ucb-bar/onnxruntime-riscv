// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace systolic {

template <typename T1, typename T2, typename T3>
class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  ConvAttributes conv_attrs_;
};

template<typename T>
inline void GemmlowpDebug(const T* lhs_data, const T* rhs_data, T* result_data,
                        const int lhs_offset, const int rhs_offset, const int result_offset,
                        int m, int n, int k, int32_t int_multiplier, int32_t right_shift, const int32_t* bias) {

  printf("lhs matrix\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      printf("%d ", (int) lhs_data[i * k + j]);
    }
    printf("\n");
  }

  printf("rhs matrix\n");
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      printf("%d ", (int) rhs_data[i * n + j]);
    }
    printf("\n");
  }

  ORT_UNUSED_PARAMETER(result_data);
  // printf("out matrix\n");
  // for (int i = 0; i < m; i++) {
  //   for (int j = 0; j < n; j++) {
  //     printf("%d ", (int) result_data[i * n + j]);
  //   }
  //   printf("\n");
  // }

  printf("m, n, k: %d %d %d\n", m, n, k);
  printf("lhs_offset: %d\n", lhs_offset);
  printf("rhs_offset: %d\n", rhs_offset);
  printf("result_offset: %d\n", result_offset);

  printf("int_multiplier: %d\n", int_multiplier);
  printf("right_shift: %d\n", right_shift);
  if (bias) {
    printf("bias:\n");
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        printf("%d ", bias[i*n + j]);
      }
      printf("\n");
    }
  }
  
}


} // namespace systolic
}  // namespace onnxruntime
