#ifndef SYSTOLIC_PARAMS_H
#define SYSTOLIC_PARAMS_H

#include <stdint.h>
#include <limits.h>

// Dimension of the systolic array
// Should be tileColumns*meshColumns
#define DIM 16
#define ADDR_LEN 32
#define BANK_NUM 4
// Unforunately, using sizeof in a macro is problematic, so we use 1 instead of
// sizeof(elem_t) and 4 instead of sizeof(acc_t)
#define BANK_ROWS (256 * 1024 / (BANK_NUM * DIM * 1))
#define ACC_ROWS (64 * 1024 / (DIM * 4))
#define MAX_BYTES 64
#define MAX_BLOCK_LEN (MAX_BYTES / (DIM * 1))
#define MAX_BLOCK_LEN_ACC (MAX_BYTES / (DIM * 4))

#define MAX_Q_LEN 256

// Datatype of the systolic array
typedef int8_t elem_t;
elem_t elem_t_max = SCHAR_MAX;
elem_t elem_t_min = SCHAR_MIN;
typedef int32_t acc_t;

#define row_align(blocks) __attribute__((aligned(blocks * DIM * sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks * DIM * sizeof(acc_t))))

#endif  // SYSTOLIC_PARAMS_H
