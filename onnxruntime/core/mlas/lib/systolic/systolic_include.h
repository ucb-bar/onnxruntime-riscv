// See LICENSE for license details.

#ifndef SRC_MAIN_C_SYSTOLIC_H
#define SRC_MAIN_C_SYSTOLIC_H

//#define ENABLE_SYSTOLIC_DEBUG

#ifdef ENABLE_SYSTOLIC_DEBUG
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...)
#endif

#include <stdint.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
// TODO use stdbool.h as well

#include "systolic_params.h"

// Accelerator interface
#include "xcustom.h"

#define k_CONFIG 0
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2

#define XCUSTOM_ACC 3

#define GARBAGE_ADDR ((uint64_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define RELU6 2

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// mvin and mvout
#define matmul_mvin(dram_addr, spad_addr) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)1 << ADDR_LEN) | (spad_addr), k_MVIN)

#define matmul_block_mvin(dram_addr, spad_addr, len) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, ((uint64_t)(len) << ADDR_LEN) | (spad_addr), k_MVIN)

#define matmul_mvout(dram_addr, spad_addr) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, dram_addr, spad_addr, k_MVOUT)

// compute
#define matmul_compute_preloaded(A, BD) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, BD, k_COMPUTE_PRELOADED)

#define matmul_compute_accumulated(A, BD) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, A, BD, k_COMPUTE_ACCUMULATE)

// preload
#define matmul_preload(BD, C) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, BD, C, k_PRELOAD)

#define matmul_preload_zeros(C) \
  matmul_preload(GARBAGE_ADDR, C)

// config
#define matmul_config_ex(mode, act, sys_shift, acc_shift, relu6_shift) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(acc_shift) << 32) | ((act) << 3) | ((mode) << 2) | CONFIG_EX, ((uint64_t)(relu6_shift) << 32) | (sys_shift), k_CONFIG)

#define matmul_config_ld(stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_LD, stride, k_CONFIG)

#define matmul_config_st(stride) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, CONFIG_ST, stride, k_CONFIG)

// flush
#define matmul_flush(skip) \
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, skip, 0, k_FLUSH)

// fence
#define matmul_fence() asm volatile("fence")

// A statically allocated scratch array. We'll just use this as a generic byte buffer without regard to the type  (so we can handle D)
int32_t padding_scrach_internal[MAX_BLOCK_LEN * DIM * 16];
void* padding_scratch = (void*)&padding_scrach_internal[0];
elem_t out_scratch[DIM][DIM];

/**
 * Pad a given matrix of dimensinos `copy_amnt_down`x`copy_amnt_across` to a 16xREQ_ROW_DIM matrix
 */
template <typename T>
void pad_matrix(const T* in, size_t stride, int copy_amnt_down, int copy_amnt_across, int required_row_dim) {
  assert(copy_amnt_down <= 16 && copy_amnt_across <= MAX_BLOCK_LEN * DIM);
  LOG("copyamnt down/across %d %d\n", copy_amnt_down, copy_amnt_across);
  LOG("required row_dim %d\n", required_row_dim);
  for (int i = 0; i < copy_amnt_down; i++) {
    std::copy(in, in + copy_amnt_across, (T*)padding_scratch + (required_row_dim)*i);
    std::fill((T*)padding_scratch + (required_row_dim)*i + copy_amnt_across, (T*)padding_scratch + (required_row_dim)*i + required_row_dim, 0);
    in += stride;
  }
  // We only ever load 16 vertically
  for (int i = copy_amnt_down; i < 16; i++) {
    std::fill((T*)padding_scratch + (required_row_dim)*i, (T*)padding_scratch + (required_row_dim)*i + required_row_dim, 0);
  }
}

template <typename T>
void load_padded_matrix(size_t col_block_index, size_t col_tile_dim, size_t col_padding,
                        size_t row_block_index, size_t row_tile_dim, size_t row_padding,
                        size_t load_blocks, size_t stride, const T* dram_addr, uint32_t sp_addr) {
  int blocks = row_block_index + load_blocks <= row_tile_dim ? load_blocks : row_tile_dim - row_block_index;
  bool padHorizontal = false;
  if (row_padding != 0 && row_block_index + blocks >= row_tile_dim) {  // We  will be loading the last column
    blocks = blocks - 1;
    padHorizontal = true;
  }

  // Split into cases based on whether or not we are amongst the bottom-most row and need to pad along I
  // If we have padding in I, we are guaranteed that it will be at i = I - 1 since we only pad at most 15
  if (col_padding == 0 || col_block_index + 1 < col_tile_dim) {
    if (blocks != 0) {
      LOG("Loading %d blocks without padding\n", blocks);
      // Load as much as we can before needing to pad
      LOG("ADDRS %lld %lld %lld\n", dram_addr, sp_addr, blocks);
      matmul_config_ld(stride * sizeof(T));
      matmul_block_mvin(dram_addr, sp_addr, blocks);
    }
    if (padHorizontal) {
      LOG("Padding horizontal\n");
      pad_matrix(dram_addr + DIM * blocks, stride, /*copy_amnt_down=*/DIM, /*copy_amnt_across= */ DIM - row_padding, /*required_row_dim= */ 16);
      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
          LOG("%d ", ((T*)padding_scratch)[i * 16 + j]);
        }
        LOG("\n");
      }
      matmul_config_ld(16 * sizeof(T));  //A 16x16 load
      matmul_block_mvin(padding_scratch, sp_addr + DIM * blocks, 1);
    }

  } else {  // We are trying to load the last row that needs padding
    if (blocks != 0) {
      pad_matrix(dram_addr, stride, /*copy_amnt_down= */ DIM - col_padding, /*copy_amnt_across= */ blocks * DIM, /*required_row_dim= */ blocks * DIM);
      LOG("Padding A vertical\n");
      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < blocks * DIM; j++) {
          LOG("%d ", ((T*)padding_scratch)[i * 16 + j]);
        }
        LOG("\n");
      }
      matmul_config_ld(blocks * DIM * sizeof(T));
      matmul_block_mvin(padding_scratch, sp_addr, blocks);
    }
    if (padHorizontal) {
      pad_matrix(dram_addr + DIM * blocks, stride, /*copy_amnt_down= */ DIM - col_padding, /*copy_amnt_across= */ DIM - row_padding, /*required_row_dim= */ 16);
      LOG("Padding vertical + horizontal\n");
      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
          LOG("%d ", ((T*)padding_scratch)[i * 16 + j]);
        }
        LOG("\n");
      }
      matmul_config_ld(16 * sizeof(T));
      matmul_block_mvin(padding_scratch, sp_addr + DIM * blocks, 1);
    }
  }
}

void move_out_padded_matrix(size_t I_PADDING, size_t TILE_I_DIM, /*index into i tile= */ size_t i,
                            size_t J_PADDING, size_t TILE_J_DIM, /*index into j tile= */ size_t j,
                            elem_t* C, const uint32_t C_sp_addr_start, size_t C_stride) {
  if ((I_PADDING == 0 || i != TILE_I_DIM - 1) && (J_PADDING == 0 || j != TILE_J_DIM - 1)) {
    elem_t* const C_dram_addr = C + (i * C_stride + j) * DIM;
    const uint32_t C_sp_addr = C_sp_addr_start + (i * TILE_J_DIM + j) * DIM;
    matmul_config_st(C_stride * sizeof(elem_t));
    matmul_mvout(C_dram_addr, C_sp_addr);

  } else if (I_PADDING != 0 && i == TILE_I_DIM - 1 && (J_PADDING == 0 || j != TILE_J_DIM - 1)) {
    LOG("Move out padded i\n");
    elem_t* C_dram_addr = C + (i * C_stride + j) * DIM;
    const uint32_t C_sp_addr = C_sp_addr_start + (i * TILE_J_DIM + j) * DIM;
    matmul_config_st(DIM * sizeof(elem_t));
    matmul_mvout(&out_scratch[0][0], C_sp_addr);

    for (size_t i = 0; i < DIM - I_PADDING; i++) {
      std::copy(&out_scratch[i][0], &out_scratch[i][0] + DIM, C_dram_addr);
      C_dram_addr += C_stride;
    }

  } else if (J_PADDING != 0 && j == TILE_J_DIM - 1 && (I_PADDING == 0 || i != TILE_I_DIM - 1)) {
    LOG("Move out padded j\n");
    const uint32_t C_sp_addr = C_sp_addr_start + (i * TILE_J_DIM + j) * DIM;
    matmul_config_st(DIM * sizeof(elem_t));
    matmul_mvout(&out_scratch[0][0], C_sp_addr);

    elem_t* C_dram_addr = C + (i * C_stride + j) * DIM;
    for (int i = 0; i < DIM; i++) {
      std::copy(&out_scratch[i][0], &out_scratch[i][0] + DIM - J_PADDING, C_dram_addr);
      C_dram_addr += C_stride;
    }

  } else {
    LOG("Move out padded i & j\n");
    const uint32_t C_sp_addr = C_sp_addr_start + (i * TILE_J_DIM + j) * DIM;
    matmul_config_st(DIM * sizeof(elem_t));
    matmul_mvout(&out_scratch[0][0], C_sp_addr);

    elem_t* C_dram_addr = C + (i * C_stride + j) * DIM;
    for (size_t i = 0; i < DIM - I_PADDING; i++) {
      std::copy(&out_scratch[i][0], &out_scratch[i][0] + DIM - J_PADDING, C_dram_addr);
      C_dram_addr += C_stride;
    }
  }
}

// Tiling functions
static void sp_tiled_matmul_os(const elem_t* A, const elem_t* B, const void* D, elem_t* C,
                               size_t TILE_I_DIM, size_t TILE_J_DIM, size_t TILE_K_DIM,
                               size_t I_PADDING, size_t J_PADDING, size_t K_PADDING,
                               size_t A_stride, size_t B_stride, size_t D_stride, size_t C_stride,
                               int no_bias, int full_bias_width) {
  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS / 2;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN - 1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN - 2);

  const int A_blocks = TILE_K_DIM <= MAX_BLOCK_LEN ? TILE_K_DIM : MAX_BLOCK_LEN;
  const int B_blocks = TILE_J_DIM <= MAX_BLOCK_LEN ? TILE_J_DIM : MAX_BLOCK_LEN;
  const size_t D_blocks_max = full_bias_width ? MAX_BLOCK_LEN_ACC : MAX_BLOCK_LEN;
  const int D_blocks = TILE_J_DIM <= D_blocks_max ? TILE_J_DIM : D_blocks_max;

  const int sizeof_bias = full_bias_width ? sizeof(acc_t) : sizeof(elem_t);

  // Move-in D
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < TILE_I_DIM; i++) {
      for (size_t j = 0; j < TILE_J_DIM; j++) {
        void* D_dram_addr;
        uint32_t D_sp_addr_acc = D_sp_addr_start + (i * TILE_J_DIM + j) * DIM;
        uint32_t D_sp_addr_sp = (i * TILE_J_DIM + j) * DIM;

        if (full_bias_width) {
          acc_t* const D_ = (acc_t*)D;
          D_dram_addr = (void*)(D_ + (i * D_stride + j) * DIM);
        } else {
          elem_t* const D_ = (elem_t*)D;
          D_dram_addr = (void*)(D_ + (i * D_stride + j) * DIM);
        }

        int already_moved_in = j % D_blocks != 0;

        if (!already_moved_in) {
          LOG("Loading in D\n");
          if (full_bias_width) {
            load_padded_matrix(i, TILE_I_DIM, I_PADDING,
                               j, TILE_J_DIM, J_PADDING,
                               D_blocks, D_stride, (acc_t*)D_dram_addr, D_sp_addr_acc);
          } else {
            load_padded_matrix(i, TILE_I_DIM, I_PADDING,
                               j, TILE_J_DIM, J_PADDING,
                               D_blocks, D_stride, (elem_t*)D_dram_addr, D_sp_addr_sp);
          }
        }

        matmul_config_ld(D_stride * sizeof_bias);
        if (!full_bias_width) {
          matmul_preload(D_sp_addr_sp, D_sp_addr_acc);
          matmul_compute_preloaded(GARBAGE_ADDR, GARBAGE_ADDR);
        }
      }
    }
  }

  for (size_t i = 0; i < TILE_I_DIM; i++) {
    for (size_t j = 0; j < TILE_J_DIM; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i * TILE_J_DIM + j) * DIM;

      for (size_t k = 0; k < TILE_K_DIM; k++) {
        // LOG("  i: %u, j: %u, k: %u\n", i, j, k);

        const elem_t* const A_dram_addr = A + (i * A_stride + k) * DIM;
        const elem_t* const B_dram_addr = B + (k * B_stride + j) * DIM;

        const uint32_t A_sp_addr = A_sp_addr_start + (i * TILE_K_DIM + k) * DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k * TILE_J_DIM + j) * DIM;

        // Move-in A and B
        {
          // LOG("    Enter mvin\n");

          int A_already_moved_in = j != 0 || k % A_blocks != 0;
          int B_already_moved_in = i != 0 || j % B_blocks != 0;

          if (!A_already_moved_in) {
            LOG("Moving in A\n");
            load_padded_matrix(i, TILE_I_DIM, I_PADDING,
                               k, TILE_K_DIM, K_PADDING,
                               A_blocks, A_stride, A_dram_addr, A_sp_addr);
          }

          if (!B_already_moved_in) {
            LOG("Moving in B\n");
            load_padded_matrix(k, TILE_K_DIM, K_PADDING,
                               j, TILE_J_DIM, J_PADDING,
                               B_blocks, B_stride, B_dram_addr, B_sp_addr);
          }

          // LOG("    Exit mvin\n");
        }

        // Compute
        {
          // LOG("    Enter compute\n");
          uint32_t out_sp_addr = k == TILE_K_DIM - 1 ? C_sp_addr : GARBAGE_ADDR;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == TILE_K_DIM - 1;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN - 2));
          }

          matmul_preload(GARBAGE_ADDR, out_sp_addr);

          if (k == 0) {  // First iteration
            matmul_compute_preloaded(A_sp_addr, B_sp_addr);
          } else {  // All other iterations
            matmul_compute_accumulated(A_sp_addr, B_sp_addr);
          }

          // LOG("    Exit compute\n");
        }
      }
    }
  }

  // LOG("Exit main inner loop\n");

  // TODO this should be overlapped with the next "Move-in D"
  // Move-out C
  if (C != NULL) {
    LOG("  Enter mvout loop\n");
    for (size_t i = 0; i < TILE_I_DIM; i++) {
      for (size_t j = 0; j < TILE_J_DIM; j++) {
        LOG("    i: %lu, j: %lu\n", i, j);
        move_out_padded_matrix(I_PADDING, TILE_I_DIM, i,
                               J_PADDING, TILE_J_DIM, j,
                               C, C_sp_addr_start, C_stride);
      }
    }
    // LOG("  Exit mvout loop\n");
  }
}

static void sp_tiled_matmul_ws(const elem_t* A, const elem_t* B, const void* D, elem_t* C,
                               size_t TILE_I_DIM, size_t TILE_J_DIM, size_t TILE_K_DIM,
                               size_t I_PADDING, size_t J_PADDING, size_t K_PADDING,
                               size_t A_stride, size_t B_stride, size_t D_stride, size_t C_stride,
                               int no_bias, int full_bias_width) {
  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS / 2;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN - 1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN - 2);

  const int A_blocks = TILE_K_DIM <= MAX_BLOCK_LEN ? TILE_K_DIM : MAX_BLOCK_LEN;
  const int B_blocks = TILE_J_DIM <= MAX_BLOCK_LEN ? TILE_J_DIM : MAX_BLOCK_LEN;
  const size_t D_blocks_max = full_bias_width ? MAX_BLOCK_LEN_ACC : MAX_BLOCK_LEN;
  const int D_blocks = TILE_J_DIM <= D_blocks_max ? TILE_J_DIM : D_blocks_max;

  const int sizeof_bias = full_bias_width ? sizeof(acc_t) : sizeof(elem_t);

  // Move-in D
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < TILE_I_DIM; i++) {
      for (size_t j = 0; j < TILE_J_DIM; j++) {
        void* D_dram_addr;
        uint32_t D_sp_addr_acc = D_sp_addr_start + (i * TILE_J_DIM + j) * DIM;
        uint32_t D_sp_addr_sp = (i * TILE_J_DIM + j) * DIM;

        if (full_bias_width) {
          acc_t* const D_ = (acc_t*)D;
          D_dram_addr = (void*)(D_ + (i * D_stride + j) * DIM);
        } else {
          elem_t* const D_ = (elem_t*)D;
          D_dram_addr = (void*)(D_ + (i * D_stride + j) * DIM);
        }

        int already_moved_in = j % D_blocks != 0;

        if (!already_moved_in) {
          LOG("Loading in D\n");
          if (full_bias_width) {
            load_padded_matrix(i, TILE_I_DIM, I_PADDING,
                               j, TILE_J_DIM, J_PADDING,
                               D_blocks, D_stride, (acc_t*)D_dram_addr, D_sp_addr_acc);
          } else {
            load_padded_matrix(i, TILE_I_DIM, I_PADDING,
                               j, TILE_J_DIM, J_PADDING,
                               D_blocks, D_stride, (elem_t*)D_dram_addr, D_sp_addr_sp);
          }
        }

        matmul_config_ld(D_stride * sizeof_bias);
        if (!full_bias_width) {
          matmul_preload(GARBAGE_ADDR, D_sp_addr_acc);
          matmul_compute_preloaded(GARBAGE_ADDR, D_sp_addr_sp);
        }
      }
    }
  }

  for (size_t j = 0; j < TILE_J_DIM; j++) {
    for (size_t k = 0; k < TILE_K_DIM; k++) {
      const uint32_t B_sp_addr = B_sp_addr_start + (k * TILE_J_DIM + j) * DIM;

      for (size_t i = 0; i < TILE_I_DIM; i++) {
        // LOG("  i: %u, j: %u, k: %u\n", i, j, k);

        const elem_t* const A_dram_addr = A + (i * A_stride + k) * DIM;
        const elem_t* const B_dram_addr = B + (k * B_stride + j) * DIM;

        const uint32_t A_sp_addr = A_sp_addr_start + (i * TILE_K_DIM + k) * DIM;
        const uint32_t C_sp_addr = C_sp_addr_start + (i * TILE_J_DIM + j) * DIM;

        // Move-in A and B
        {
          // LOG("    Enter mvin\n");

          int A_already_moved_in = j != 0 || k % A_blocks != 0;
          int B_already_moved_in = i != 0 || j % B_blocks != 0;

          if (!A_already_moved_in) {
            LOG("Moving in A\n");
            load_padded_matrix(i, TILE_I_DIM, I_PADDING,
                               k, TILE_K_DIM, K_PADDING,
                               A_blocks, A_stride, A_dram_addr, A_sp_addr);
          }

          if (!B_already_moved_in) {
            LOG("Moving in B\n");
            load_padded_matrix(k, TILE_K_DIM, K_PADDING,
                               j, TILE_J_DIM, J_PADDING,
                               B_blocks, B_stride, B_dram_addr, B_sp_addr);
          }

          // LOG("    Exit mvin\n");
        }

        // Compute
        {
          // LOG("    Enter compute\n");
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;

          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN - 2));
          }

          matmul_preload(pre_sp_addr, out_sp_addr);

          if (i == 0) {  // First iteration
            matmul_compute_preloaded(A_sp_addr, GARBAGE_ADDR);
          } else {  // All other iterations
            matmul_compute_accumulated(A_sp_addr, GARBAGE_ADDR);
          }
          // LOG("    Exit compute\n");
        }
      }
    }
  }

  // LOG("Exit main inner loop\n");

  // TODO this should be overlapped with the next "Move-in D"
  // Move-out C
  if (C != NULL) {
    LOG("  Enter mvout loop\n");
    for (size_t i = 0; i < TILE_I_DIM; i++) {
      for (size_t j = 0; j < TILE_J_DIM; j++) {
        LOG("    i: %lu, j: %lu\n", i, j);
        move_out_padded_matrix(I_PADDING, TILE_I_DIM, i,
                               J_PADDING, TILE_J_DIM, j,
                               C, C_sp_addr_start, C_stride);
      }
    }
    // LOG("  Exit mvout loop\n");
  }
}

/**
 * elem_t A[DIM_I][DIM_K]
 * elem_t B[DIM_K][DIM_J]
 * void* D
 * elem_t C[DIM_I][DIM_J]
 */
static void tiled_matmul_acc(size_t DIM_I, size_t DIM_J, size_t DIM_K,
                             size_t DIM_I_PADDED, size_t DIM_J_PADDED, size_t DIM_K_PADDED,
                             size_t strideA,
                             size_t strideB,
                             size_t strideD,
                             size_t strideC,
                             const elem_t* A, const elem_t* B, const void* D,
                             elem_t* C, size_t TILE_I, size_t TILE_J, size_t TILE_K,
                             int act, int shift, int relu6_shift, int full_bias_width, bool is_weight_stationary) {
  LOG("Tiling: %d %d %d\n", TILE_I, TILE_J, TILE_K);
  LOG("Padding: %d %d %d\n", DIM_I_PADDED, DIM_J_PADDED, DIM_K_PADDED);

  const size_t I0 = DIM_I_PADDED / (TILE_I * DIM);
  const size_t J0 = DIM_J_PADDED / (TILE_J * DIM);
  const size_t K0 = DIM_K_PADDED / (TILE_K * DIM);

  const int no_bias = D == NULL;

  if (no_bias) {
    // D = (acc_t (*)[DIM_J]) 1; // Dummy address which isn't NULL
    D = (elem_t*)1;  // Dummy address which isn't NULL
  }

  matmul_config_ex(is_weight_stationary, act, 0, shift, relu6_shift);
  matmul_config_st(DIM_J * sizeof(elem_t));

  LOG("I0 J0 K0 %d %d %d\n", I0, J0, K0);
  LOG("strides = %d %d %d\n", strideA, strideB, strideC);

  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {
        LOG("i0: %lu, j0: %lu, k0: %lu\n", i0, j0, k0);

        LOG("first_mvin? %d\n", i0 == 0 && j0 == 0 && k0 == 0);
        LOG("last_mvout? %d\n", (i0 == I0 - 1) && (j0 == J0 - 1) && (k0 == K0 - 1));

        // acc_t * pre = k0 == 0 ? &D[i0*TILE_I*DIM][j0*TILE_J*DIM] : NULL;
        void* pre;
        if (k0 != 0) {
          pre = NULL;
        } else if (full_bias_width) {
          pre = &(((acc_t*)D)[i0 * TILE_I * DIM * strideD + j0 * TILE_J * DIM]);
        } else {
          pre = &(((elem_t*)D)[i0 * TILE_I * DIM * strideD + j0 * TILE_J * DIM]);
        }

        elem_t* out = k0 == K0 - 1 ? &C[i0 * TILE_I * DIM * strideC + j0 * TILE_J * DIM] : NULL;

        LOG("A index: %d %d\n", i0 * TILE_I * DIM, k0 * TILE_K * DIM);
        LOG("B index: %d %d\n", k0 * TILE_K * DIM, j0 * TILE_J * DIM);

        LOG("Starting A address: %d %d\n", i0 * TILE_I * DIM, k0 * TILE_K * DIM);
        LOG("Starting B address: %d %d\n", k0 * TILE_K * DIM, j0 * TILE_J * DIM);

        if (is_weight_stationary) {
          sp_tiled_matmul_ws(&A[i0 * TILE_I * DIM * strideA + k0 * TILE_K * DIM],
                             &B[k0 * TILE_K * DIM * strideB + j0 * TILE_J * DIM],
                             pre, out,
                             TILE_I, TILE_J, TILE_K,
                             i0 == I0 - 1 ? DIM_I_PADDED - DIM_I : 0,
                             j0 == J0 - 1 ? DIM_J_PADDED - DIM_J : 0,
                             k0 == K0 - 1 ? DIM_K_PADDED - DIM_K : 0,
                             strideA, strideB, strideD, strideC,
                             no_bias, full_bias_width);
        } else {
          sp_tiled_matmul_os(&A[i0 * TILE_I * DIM * strideA + k0 * TILE_K * DIM],
                             &B[k0 * TILE_K * DIM * strideB + j0 * TILE_J * DIM],
                             pre, out,
                             TILE_I, TILE_J, TILE_K,
                             i0 == I0 - 1 ? DIM_I_PADDED - DIM_I : 0,
                             j0 == J0 - 1 ? DIM_J_PADDED - DIM_J : 0,
                             k0 == K0 - 1 ? DIM_K_PADDED - DIM_K : 0,
                             strideA, strideB, strideD, strideC,
                             no_bias, full_bias_width);
        }
      }

  matmul_fence();
}

#define ROUNDING_RIGHT_SHIFT(x, shift)                      \
  ({ ((x) >> (shift)) +                                     \
         (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
          ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1))); })

inline elem_t saturate(acc_t num, const void* D, int pos, int act, int shift, int relu6_shift, int full_bias_width) {
  const int no_bias = D == NULL;
  acc_t result = num;
  if (!no_bias && full_bias_width) {
    result += ((acc_t*)D)[pos];
  } else if (!no_bias && !full_bias_width) {
    result += ((elem_t*)D)[pos];
  }

  // Scale value down and round it
  result = ROUNDING_RIGHT_SHIFT(result, shift);

  // Clip result
  result = result > elem_t_max ? elem_t_max : (result < elem_t_min ? elem_t_min : result);

  // Apply activation function
  if (act == RELU) {
    result = result < 0 ? 0 : result;
  } else if (act == RELU6) {
    int max = 6 << relu6_shift;
    result = result < 0 ? 0 : (result > max ? max : result);
  }
  return result;
}

static void matmul_cpu(size_t DIM_I, size_t DIM_J, size_t DIM_K,
                       size_t strideA,
                       size_t strideB,
                       size_t strideD,
                       size_t strideC,
                       const elem_t* A, const elem_t* B, const void* D,
                       elem_t* C,
                       int act, int shift, int relu6_shift, int full_bias_width) {
  if (DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {
        acc_t result[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] += A[i * strideA + k] * B[k * strideB + j];
          result[0][1] += A[i * strideA + k] * B[k * strideB + j + 1];
          result[0][2] += A[i * strideA + k] * B[k * strideB + j + 2];
          result[0][3] += A[i * strideA + k] * B[k * strideB + j + 3];

          result[1][0] += A[(i + 1) * strideA + k] * B[k * strideB + j];
          result[1][1] += A[(i + 1) * strideA + k] * B[k * strideB + j + 1];
          result[1][2] += A[(i + 1) * strideA + k] * B[k * strideB + j + 2];
          result[1][3] += A[(i + 1) * strideA + k] * B[k * strideB + j + 3];

          result[2][0] += A[(i + 2) * strideA + k] * B[k * strideB + j];
          result[2][1] += A[(i + 2) * strideA + k] * B[k * strideB + j + 1];
          result[2][2] += A[(i + 2) * strideA + k] * B[k * strideB + j + 2];
          result[2][3] += A[(i + 2) * strideA + k] * B[k * strideB + j + 3];

          result[3][0] += A[(i + 3) * strideA + k] * B[k * strideB + j];
          result[3][1] += A[(i + 3) * strideA + k] * B[k * strideB + j + 1];
          result[3][2] += A[(i + 3) * strideA + k] * B[k * strideB + j + 2];
          result[3][3] += A[(i + 3) * strideA + k] * B[k * strideB + j + 3];
        }
        C[i * strideC + j] = saturate(result[0][0], D, i * strideD + j, act, shift, relu6_shift, full_bias_width);
        C[i * strideC + j + 1] = saturate(result[0][1], D, i * strideD + j + 1, act, shift, relu6_shift, full_bias_width);
        C[i * strideC + j + 2] = saturate(result[0][2], D, i * strideD + j + 2, act, shift, relu6_shift, full_bias_width);
        C[i * strideC + j + 3] = saturate(result[0][3], D, i * strideD + j + 3, act, shift, relu6_shift, full_bias_width);

        C[(i + 1) * strideC + j] = saturate(result[1][0], D, (i + 1) * strideD + j, act, shift, relu6_shift, full_bias_width);
        C[(i + 1) * strideC + j + 1] = saturate(result[1][1], D, (i + 1) * strideD + j + 1, act, shift, relu6_shift, full_bias_width);
        C[(i + 1) * strideC + j + 2] = saturate(result[1][2], D, (i + 1) * strideD + j + 2, act, shift, relu6_shift, full_bias_width);
        C[(i + 1) * strideC + j + 3] = saturate(result[1][3], D, (i + 1) * strideD + j + 3, act, shift, relu6_shift, full_bias_width);

        C[(i + 2) * strideC + j] = saturate(result[2][0], D, (i + 2) * strideD + j, act, shift, relu6_shift, full_bias_width);
        C[(i + 2) * strideC + j + 1] = saturate(result[2][1], D, (i + 2) * strideD + j + 1, act, shift, relu6_shift, full_bias_width);
        C[(i + 2) * strideC + j + 2] = saturate(result[2][2], D, (i + 2) * strideD + j + 2, act, shift, relu6_shift, full_bias_width);
        C[(i + 2) * strideC + j + 3] = saturate(result[2][3], D, (i + 2) * strideD + j + 3, act, shift, relu6_shift, full_bias_width);

        C[(i + 3) * strideC + j] = saturate(result[3][0], D, (i + 3) * strideD + j, act, shift, relu6_shift, full_bias_width);
        C[(i + 3) * strideC + j + 1] = saturate(result[3][1], D, (i + 3) * strideD + j + 1, act, shift, relu6_shift, full_bias_width);
        C[(i + 3) * strideC + j + 2] = saturate(result[3][2], D, (i + 3) * strideD + j + 2, act, shift, relu6_shift, full_bias_width);
        C[(i + 3) * strideC + j + 3] = saturate(result[3][3], D, (i + 3) * strideD + j + 3, act, shift, relu6_shift, full_bias_width);
      }
    }
  } else {
    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        acc_t result = 0;
        for (size_t k = 0; k < DIM_K; k++) {
          result += A[i * strideA + k] * B[k * strideB + j];
        }
        C[i * strideC + j] = saturate(result, D, i * strideD + j, act, shift, relu6_shift, full_bias_width);
      }
    }
  }
}

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t { CPU,
                           OS,
                           WS };

// TODO add support for non-divisible tiling factors
static size_t tiling_factor(const size_t dimension, const size_t max_tile_factor) {
  const size_t start = dimension < max_tile_factor ? dimension : max_tile_factor;

  for (size_t tile_factor = start; tile_factor >= 1; tile_factor--) {
    if (dimension % tile_factor == 0)
      return tile_factor;
  }
  return 1;  // We should never reach here anyway
}

static void __attribute__((unused)) tiled_matmul_option(size_t DIM_I, size_t DIM_J, size_t DIM_K,
                                                        size_t strideA,
                                                        size_t strideB,
                                                        size_t strideD,
                                                        size_t strideC,
                                                        const elem_t* A, const elem_t* B, const void* D,
                                                        elem_t* C,
                                                        int act, int shift, int relu6_shift, int full_bias_width,
                                                        enum tiled_matmul_type_t tiled_matmul_type) {
  switch (tiled_matmul_type) {
    case CPU:
      printf("NOTE: Using systolic CPU emulation.\n");
      break;
    case OS:
      printf("NOTE: Using systolic OS mode.\n");
      break;
    case WS:
      printf("NOTE: Using systolic WS mode.\n");
      break;
  }
  // const int partition_rows = BANK_NUM * BANK_ROWS / 2;
  // const int mats_in_partition = partition_rows / DIM;
  // const int mats_in_acc = ACC_ROWS / DIM;
  // const int max_tile_i_j = (int)sqrt(mats_in_acc);
  // const int max_tile_k = mats_in_partition / max_tile_i_j;

  // We use macros here instead of "const int" so that GCC const-folds sqrt

  size_t DIM_I_PADDED = (DIM_I + 16 - 1) / 16 * 16;
  size_t DIM_J_PADDED = (DIM_J + 16 - 1) / 16 * 16;
  size_t DIM_K_PADDED = (DIM_K + 16 - 1) / 16 * 16;
  LOG("Original DIM %d %d %d\n", DIM_I, DIM_J, DIM_K);
  LOG("Padded DIM %d %d %d\n", DIM_I_PADDED, DIM_J_PADDED, DIM_K_PADDED);

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((int)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

  LOG("Max ij,k %d %d\n", max_tile_i_j, max_tile_k);

  const size_t tile_i = tiling_factor(DIM_I_PADDED / DIM, max_tile_i_j);
  const size_t tile_j = tiling_factor(DIM_J_PADDED / DIM, max_tile_i_j);
  const size_t tile_k = tiling_factor(DIM_K_PADDED / DIM, max_tile_k);

  LOG("tile_i: %lu\n", tile_i);
  LOG("tile_j: %lu\n", tile_j);
  LOG("tile_k: %lu\n", tile_k);

  assert((DIM_I_PADDED % (tile_i * DIM) == 0) && "I dimension not divisible by tiling factor");
  assert((DIM_J_PADDED % (tile_j * DIM) == 0) && "J dimension not divisible by tiling factor");
  assert((DIM_K_PADDED % (tile_k * DIM) == 0) && "K dimension not divisible by tiling factor");

  if (tiled_matmul_type == OS) {
    tiled_matmul_acc(DIM_I, DIM_J, DIM_K,
                     DIM_I_PADDED, DIM_J_PADDED, DIM_K_PADDED,
                     strideA,
                     strideB,
                     strideD,
                     strideC,
                     A, B, D, C,
                     tile_i, tile_j, tile_k,
                     act, shift, relu6_shift, full_bias_width, OUTPUT_STATIONARY);
    // std::unique_ptr<elem_t[]> gold = std::unique_ptr<elem_t[]>(new elem_t[DIM_I * DIM_J]);
    //     matmul_cpu(DIM_I, DIM_J, DIM_K,
    //            strideA,
    //            strideB,
    //            strideD,
    //            strideC,
    //            A, B, D, gold.get(),
    //            act, shift, relu6_shift, full_bias_width);
    //   for (size_t i = 0; i < DIM_I; i++) {
    //     for (size_t j = 0; j < DIM_J; j++) {
    //       if (C[i * DIM_J + j] != gold.get()[i * DIM_J + j]) {
    //        if (DIM_I == 128 && DIM_J == 49 && DIM_K == 800) {
    //        std::ofstream myfile("dumped.txt");
    //        myfile << "shift: " << shift << "\n";
    //         myfile << "A_matrix" << "\n";
    //          for (size_t ii = 0; ii < DIM_I; ii++) {
    //            for (size_t kk = 0; kk < DIM_K; kk++) {
    //              myfile << (int) A[ii * DIM_K + kk] << " ";
    //            }
    //            myfile << "\n";
    //          }

    //          myfile << "B_matrix" << "\n";
    //          for (size_t kk = 0; kk < DIM_K; kk++) {
    //            for (size_t jj = 0; jj < DIM_J; jj++) {
    //              myfile << (int) B[kk * DIM_J + jj] << " ";
    //            }
    //            myfile << "\n";
    //          }

    //          myfile << "D_matrix" << "\n";
    //          for (size_t ii = 0; ii < DIM_I; ii++) {
    //            for (size_t jj = 0; jj < DIM_J; jj++) {
    //              myfile << (int) ((acc_t *)D)[ii * DIM_J + jj] << " ";
    //            }
    //            myfile << "\n";
    //          }

    //          myfile << "C matrix" << "\n";
    //          for (size_t ii = 0; ii < DIM_I; ii++) {
    //            for (size_t jj = 0; jj < DIM_J; jj++) {
    //              myfile << (int) C[ii * DIM_J + jj] << " ";
    //            }
    //            myfile << "\n";
    //          }

    //          myfile << "C_expected matrix" << "\n";
    //          for (size_t ii = 0; ii < DIM_I; ii++) {
    //            for (size_t jj = 0; jj < DIM_J; jj++) {
    //              myfile << (int) gold.get()[ii * DIM_J + jj] << " ";
    //            }
    //            myfile << "\n";
    //          }
    //        }
    //         printf("MISMATCH: dim %zu %zu %zu\n", DIM_I, DIM_J, DIM_K);
    //         return;
    //       }
    //     }
    //   }
    //   printf("First few values %d %d %d %d\n", gold[0], C[0], gold[1], C[1]);
  } else if (tiled_matmul_type == WS) {
    tiled_matmul_acc(DIM_I, DIM_J, DIM_K,
                     DIM_I_PADDED, DIM_J_PADDED, DIM_K_PADDED,
                     strideA,
                     strideB,
                     strideD,
                     strideC,
                     A, B, D, C,
                     tile_i, tile_j, tile_k,
                     act, shift, relu6_shift, full_bias_width, WEIGHT_STATIONARY);
  } else /*if (tiled_matmul_type == CPU)*/ {
    matmul_cpu(DIM_I, DIM_J, DIM_K,
               strideA,
               strideB,
               strideD,
               strideC,
               A, B, D, C,
               act, shift, relu6_shift, full_bias_width);
  } /* else {
        LOG("unknown tiled matrix type");
        exit(1);
    }*/

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

static void __attribute__((unused)) tiled_matmul_option(size_t DIM_I, size_t DIM_J, size_t DIM_K,
                                                        // elem_t **A, elem_t **B, acc_t D[DIM_I][DIM_J],
                                                        const elem_t* A, const elem_t* B, const void* D,
                                                        elem_t* C,
                                                        int act, int shift, int relu6_shift, int full_bias_width,
                                                        enum tiled_matmul_type_t tiled_matmul_type) {
  tiled_matmul_option(DIM_I, DIM_J, DIM_K, DIM_K, DIM_J, DIM_J, DIM_J, A, B, D, C, act, shift, relu6_shift, full_bias_width, tiled_matmul_type);
}


#endif  // SRC_MAIN_C_SYSTOLIC_H