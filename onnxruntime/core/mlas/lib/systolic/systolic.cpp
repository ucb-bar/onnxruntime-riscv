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

#define ROTATED_MATMUL_TYPE(x)

/**
 * Interally CPU is last in tiled_matmul_type_t but we want to expose CPU as accelerator mode 0
 * So just rotate everything by one
 */
inline int positive_mod(int i, int n) {
  return (i % n + n) % n;
}
inline tiled_matmul_type_t get_accelerator_mode(int mode) {
  return static_cast<tiled_matmul_type_t>(positive_mod(mode - 1, (int)CPU + 1));
}

void SystolicMultiply(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK,
                             const elem_t* in1, const elem_t* in2, elem_t* out, acc_scale_t real_multiplier, const acc_t* bias) {
  printf("Called into systolic matmul!\n");
  printf("Using accelerated matmul with dimensions (%d, %d, %d)\n", dimI, dimJ, dimK);
  tiled_matmul_auto(dimI, dimJ, dimK, in1, in2, bias, out, /*activation= */ relu,
                    real_multiplier, /*relu6_shift= */ 0, /* repeating_bias= */ 0, get_accelerator_mode(accelerator_mode));
}

void SystolicMultiply(char accelerator_mode, bool relu,
                             int dimI, int dimJ, int dimK,
                             const elem_t* in1, int strideIn1,
                             const elem_t* in2, int strideIn2,
                             elem_t* out, int strideOut,
                            acc_scale_t real_multiplier,
                             const acc_t* bias, int strideBias, bool repeating_bias) {
  printf("Called into systolic matmul!\n");
  printf("Using accelerated matmul with dimensions (%d, %d, %d)\n", dimI, dimJ, dimK);
  tiled_matmul_auto(dimI, dimJ, dimK,
                    strideIn1, strideIn2, strideBias, strideOut,
                    in1, in2, bias, out, /*activation= */ relu,
                    real_multiplier, /*relu6_shift= */ 0, /* repeating_bias= */ repeating_bias, get_accelerator_mode(accelerator_mode));
}

void SystolicAdd(char accelerator_mode __attribute__((unused)), bool relu, const int8_t* in1, float in1_scale, const int8_t* in2,
         float in2_scale __attribute__((unused)),
         int8_t* out, float out_scale, int dim) {
  printf("Called into systolic add! Relu? %d In1 scale %f, in2 scale %f, out scale %f\n", (int) relu, in1_scale, in2_scale, out_scale);
  for (int i = 0; i < dim; i++) {
    int32_t tmp1 = (int) ACC_SCALE(*in1, in1_scale/out_scale);
    int32_t tmp2 = (int) ACC_SCALE(*in2, in2_scale/out_scale);
    *out = scale_and_sat(tmp1 + tmp2, relu ? RELU : 0, 1, 0);

    out++;
    in1++;
    in2++;
  }
}

void SystolicAdd(char accelerator_mode __attribute__((unused)), bool relu, const int8_t in1, float in1_scale, const int8_t* in2,
         float in2_scale,
         int8_t* out, float out_scale, int dim) {
  printf("Called into systolic add!\n");
  for (int i = 0; i < dim; i++) {
    *out = scale_and_sat(((in1) * in1_scale + (*in2) * in2_scale)/out_scale, relu ? RELU : 0, 1, 1);
    out++;
    in2++;
  }
}

void SystolicAdd(char accelerator_mode __attribute__((unused)), bool relu, const int8_t *in1, float in1_scale, const int8_t in2,
         float in2_scale,
         int8_t* out, float out_scale, int dim) {
  printf("Called into systolic add!\n");
  for (int i = 0; i < dim; i++) {
    *out = scale_and_sat(((*in1) * in1_scale + (in2) * in2_scale)/out_scale, relu ? RELU : 0, 1, 1);
    out++;
    in1++;
  }
}

