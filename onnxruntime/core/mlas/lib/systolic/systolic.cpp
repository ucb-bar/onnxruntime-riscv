// See LICENSE for license details.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>
#include "systolic_include.h"

// inline elem_t saturate(acc_t num, int shift) {
//   const int divisor = 1 << shift;
//   acc_t abs = num > 0 ? num : -num;
//   acc_t shifted = (abs + (divisor/2)) / divisor;
//   if (num < 0)
//       num = -shifted;
//   else
//       num = shifted;

//   // Clip result
//   return num > elem_t_max ? elem_t_max : (num < elem_t_min ? elem_t_min : num);
// }

// void mymatmul(int dimI, int dimJ, int dimK, const elem_t* in1, const elem_t* in2, elem_t* out, int shift) {
// 	for (int i = 0; i < dimI; i++) {
// 		for (int j = 0; j < dimJ; j++) {
//       acc_t res = 0;
// 			for (int k = 0; k < dimK; k++) {
// 				res += in1[i * dimK + k] * in2[k * dimJ + j];
// 			}
//       out[i * dimJ + j] = saturate(res, shift);
// 		}
// 	}
// }

// void mymatmul(int dimI, int dimJ, int dimK, int striderow_in1, int striderow_in2, int striderow_out,  const elem_t* in1, const elem_t* in2, elem_t* out, bool preserve, int shift) {
// 	for (int i = 0; i < dimI; i++) {
// 		for (int j = 0; j < dimJ; j++) {
//       acc_t res = preserve ? out[i * striderow_out + j] * ( 1 << shift) : 0;
//       if (preserve && (out[i * striderow_out + j] >= elem_t_max || out[i * striderow_out + j] <= elem_t_min)) {
//         continue;
//       }
// 			for (int k = 0; k < dimK; k++) {
//         //printf("in1[%d][%d] = %d * in2[%d][%d] = %d \n", i, k, in1[i * striderow_in1 + k], k, j, in2[k * striderow_in2 + j]);
// 				res += in1[i * striderow_in1 + k] * in2[k * striderow_in2 + j];
// 			}
//       out[i * striderow_out + j] = saturate(res, shift);
//       //printf("out[%d][%d], %d\n", i, j, out[i * striderow_out + j]);
// 		}
// 	}
// }

inline int8_t saturate(int32_t num, int shift) {
    num = ROUNDING_RIGHT_SHIFT(num, shift);
    // Clip result
    return num > SCHAR_MAX ? SCHAR_MAX : (num < SCHAR_MIN ? SCHAR_MIN : num);
}

inline void mymatmul(int dimI, int dimJ, int dimK, const int8_t* in1, const int8_t* in2, int8_t* out, float real_multiplier, const int32_t* bias = nullptr) {
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

/**
 * Perform a matmul and subsequent quantization.
 * Switch between TILED_OS and TILED_CPU
 * 
 * Elements are accumulated internally into acc_t (int32) and subsequently rounded/saturated to elem_t (int8).
 * The given divisor *must* be a power of 2.
 * 
 * Note that due to Systolic limitations, if arbitrary dimension then portions of the matrix will be quantized twice so results might differ from naive CPU impl
 * Rounding behavior of Systolic currently differs from standard "round to evens" usd by numpy/ONNX
 */
void SystolicMultiplyi8i8_i8(char accelerator_mode, int dimI, int dimJ, int dimK, const elem_t* in1, const elem_t* in2, elem_t* out, int divisor, float real_multiplier, const int32_t* bias) {
  printf("Called into systolic matmul!\n");
  bool isPowerOf2 = divisor && !(divisor & (divisor - 1));
  if (!isPowerOf2) {
    throw std::runtime_error("Divisor passed to systolic matmul must be power of 2");
  }
  int shift = sizeof(int) * 8 - __builtin_clz(divisor) - 1;
  if (dimI % DIM != 0 || dimJ % DIM != 0 || dimK % DIM != 0) {
    printf("Matrix dimensions (%d, %d, %d) not multiple of systolic size. Falling back to naive CPU\n", dimI, dimJ, dimK);
    mymatmul(dimI, dimJ, dimK, in1, in2, out, real_multiplier, bias);
    return;
  }

  printf("Using accelerated matmul with dimensions (%d, %d, %d)\n", dimI, dimJ, dimK);
  tiled_matmul_option(dimI, dimJ, dimK, in1, in2, bias, out, NO_ACTIVATION, shift, /*relu6_shift= */ 0, /* full_bas_width= */ 1, static_cast<tiled_matmul_type_t>(accelerator_mode));

  // for (int i = 0; i < dimI; i++) {
  //   for (int j = 0; j < dimJ; j++) {
  //     printf("%d, ", out[i*dimJ + j]);
  //   }
  //   printf("\n");
  // }

  // Commented code handles arbitrary dimensions at cost of remultiplying
  // if (dimI > DIM && dimJ > DIM && dimK > DIM) {
  //   int maxDimI = dimI - (dimI % DIM);
  //   int maxDimJ = dimJ - (dimJ % DIM);
  //   int maxDimK = dimK - (dimK % DIM);

  //   tiled_matmul_option(maxDimI, maxDimJ, maxDimK, dimK, dimJ, 0, dimJ, in1, in2, NULL, out, NO_ACTIVATION, shift, 0, 0, CPU);

  //   mymatmul(maxDimI, maxDimJ, dimK - maxDimK, dimK, dimJ, dimJ, in1 + (maxDimK), in2 + dimJ*(maxDimK), out, true, shift);
  //   mymatmul(maxDimI, dimJ - maxDimJ, dimK, dimK, dimJ, dimJ, in1, in2 + maxDimJ, out + maxDimJ, false, shift);
  //   mymatmul(dimI - maxDimI, dimJ, dimK, in1 + dimK*(maxDimI), in2, out + dimJ*(maxDimI), shift);
  // } else {
  //   return mymatmul(dimI, dimJ, dimK, in1, in2, out, shift);
  // }
}

// void NaiveCPUMultiplyi8i8_i8(int dimI, int dimJ, int dimK, const elem_t* in1, const elem_t* in2, elem_t* out, int divisor) {
//   bool isPowerOf2 = divisor && !(divisor & (divisor - 1));
//   if (!isPowerOf2) {
//     throw std::runtime_error("Divisor passed to systolic matmul must be power of 2");
//   }
//   int shift = sizeof(int)*8 - __builtin_clz(divisor) - 1;
//   return mymatmul(dimI, dimJ, dimK, in1, in2, out, shift);
// }
