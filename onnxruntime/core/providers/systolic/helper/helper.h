#pragma once

inline int nearestPowerOfTwo(int n)
{
    if (n == 0) {
      return 1;
    }
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
  ORT_UNUSED_PARAMETER(bias);
  // if (bias) {
  //   printf("bias:\n");
  //   for (int i = 0; i < m; i++) {
  //     for (int j = 0; j < n; j++) {
  //       printf("%d ", bias[i*n + j]);
  //     }
  //     printf("\n");
  //   }
  // }
  
}



template<typename T>
inline void GemmlowpDebug(int m, int n, int k,
                            const T* lhs_data, int strideA,
                            const T* rhs_data, int strideB,
                            T* out, int strideOut,
                            float scale,
                            const int32_t* bias, int strideBias) {

  ORT_UNUSED_PARAMETER(bias);
  printf("lhs matrix\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      printf("%d ", (int) lhs_data[i * strideA + j]);
    }
    printf("\n");
  }

  printf("rhs matrix\n");
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      printf("%d ", (int) rhs_data[i * strideB + j]);
    }
    printf("\n");
  }

  ORT_UNUSED_PARAMETER(strideBias);
  ORT_UNUSED_PARAMETER(out);
  
  printf("out matrix\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%d ", (int) out[i * strideOut + j]);
    }
    printf("\n");
  }

  printf("m, n, k: %d %d %d\n", m, n, k);
  printf("scale: %f\n", scale);
  if (bias) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%d ", (int) bias[i * strideBias + j]);
    }
    printf("\n");
  }
  }
}