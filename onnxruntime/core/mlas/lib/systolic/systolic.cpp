// See LICENSE for license details.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>
#include "systolic_include.h"

/**
 * Perform a matmul and subsequent quantization.
 * Switch between TILED_OS and TILED_CPU
 * 
 * Elements are accumulated internally into acc_t (int32) and subsequently rounded/saturated to elem_t (int8).
 * The given divisor *must* be a power of 2.
 */
void SystolicMultiplyi8i8_i8(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK,
                                    const elem_t* in1, const elem_t* in2, elem_t* out, int divisor, __attribute__((unused)) float real_multiplier, const int32_t* bias) {
  printf("Called into systolic matmul!\n");
  bool isPowerOf2 = divisor && !(divisor & (divisor - 1));
  if (!isPowerOf2) {
    throw std::runtime_error("Divisor passed to systolic matmul must be power of 2");
  }
  int shift = sizeof(int) * 8 - __builtin_clz(divisor) - 1;
  // if (dimI % DIM != 0 || dimJ % DIM != 0 || dimK % DIM != 0) {
  //   printf("Matrix dimensions (%d, %d, %d) not multiple of systolic size. Falling back to naive CPU\n", dimI, dimJ, dimK);
  //   mymatmul(dimI, dimJ, dimK, in1, in2, out, real_multiplier, relu, bias);
  //   return;
  // }

  printf("Using accelerated matmul with dimensions (%d, %d, %d)\n", dimI, dimJ, dimK);
  tiled_matmul_option(dimI, dimJ, dimK, in1, in2, bias, out, /*activation= */ relu,
                      shift, /*relu6_shift= */ 0, /* full_bas_width= */ 1, static_cast<tiled_matmul_type_t>(accelerator_mode));
}
