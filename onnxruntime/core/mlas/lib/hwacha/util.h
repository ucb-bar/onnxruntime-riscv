#ifndef util_h
#define util_h

#include <stdint.h>




#define MAX(a, b)  ((a>b)?a:b)
void hwacha_init();
void setvcfg(int nd, int nw, int nh, int np);
int setvlen(int vlen);

#define ROUNDING_RIGHT_SHIFT(x, shift)                      \
  ({ ((x) >> (shift)) +                                     \
         (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
          ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1))); })



// int rdcycle();
// int rdinstret();
// void* safe_malloc(int size);
// void printfloatmatrix(int channels, int width, int height, float* M);
// void printintmatrix(int channels, int width, int height, int* M);
// void printint16matrix(int channels, int width, int height, int16_t* M);
// void fill_seq_32(float* p, int n, int mode);
// void fill_seq_16(int16_t* p, int n, int mode);
// void setvcfg(int nd, int nw, int nh, int np);

//
// void memcpy_16(int16_t* src, int16_t* dest, int len);
// void memcpy_32(float* src, float* dest, int len);
//
// void cvt_32_16(float* src, int16_t* dest, int len);
// void cvt_16_32(int16_t* src, float* dest, int len);
//
// void gather_16(const int* id, const int16_t* src, int16_t* dest, int len);
// void gather_32(const int* id, const float* src, float* dest, int len);
// void fill_16(int N, float ALPHA, int16_t* X);
// void fill_32(int N, float ALPHA, float* X);
// void normalize_16(int16_t *x, int16_t *mean, int16_t *variance, int filters, int spatial);
// void normalize_32(float *x, float *mean, float *variance, int filters, int spatial);
// void scale_16(int16_t* x, int16_t scale, int size);
// void scale_32(float* x, float scale, int size);
// void add_16(int16_t* x, int16_t y, int size);
// void add_32(float* x, float y, int size);
//
// void square_32(int N, float* x, float* dest);
//
// void axpy_32(int N, float A, float* X, float* Y);
//
// void mul_32(int N, float* X, float* Y);
#endif