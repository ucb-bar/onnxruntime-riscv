#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include "util.h"
#include <stdio.h>

#define VRU_ENABLE

#ifdef VRU_ENABLE
// because gcc complains about shifting without L
#define VRU_SWITCH 0x8000000000000000
#else
#define VRU_SWITCH 0x0
#endif

#define VCFG(nvvd, nvvw, nvvh, nvp) \
  (((nvvd) & 0x1ff) | \
  (((nvp) & 0x1f) << 9) | \
  (((nvvw) & 0x1ff) << 14) | \
  (((nvvh) & 0x1ff) << 23) | \
  (VRU_SWITCH))

void hwacha_init() {
  asm volatile ("lw t0, vsetvlen" : : : "t0");
  //asm volatile ("lw t0, vtest_avi" : : : "t0");

}

// void setvcfg(int nd, int nw, int nh, int np) {
//     int cfg = VCFG(nd, nw, nh, np);
//     asm volatile ("vsetcfg %0"
//                   :
//                   : "r" (cfg));
// }


// int __attribute__((optimize("O0"))) rdcycle() {
//     int out = 0;
//     asm("rdcycle %0" : "=r" (out));
//     return out;
// }
//
// int __attribute__((optimize("O0"))) rdinstret() {
//     int out = 0;
//     asm("rdinstret %0" : "=r" (out));
//     return out;
// }
//
// void* __attribute__((optimize("O0"))) safe_malloc(int size) {
//     void* ptr = memalign(16, size);
//     for (int i = 0; i < size / 4; i += (1 << 10)) {
//         ((int*)ptr)[i] = 1;
//     }
//     return ptr;
// }
//
// void printfloatmatrix(int channels, int width, int height, float* M) {
//     printf("\n");
//     for (int c = 0; c < channels; c++) {
//         for (int i = 0; i < height; i++) {
//             for (int j = 0; j < width; j++) {
//                 printf("%.3f\t", M[c*height*width+i*width+j]);
//             }
//             printf("\n");
//         }
//         printf("-----\n");
//     }
// }
// void printintmatrix(int channels, int width, int height, int* M) {
//     printf("\n");
//     for (int c = 0; c < channels; c++) {
//         for (int i = 0; i < height; i++) {
//             for (int j = 0; j < width; j++) {
//                 printf("%d\t", M[c*height*width+i*width+j]);
//             }
//             printf("\n");
//         }
//         printf("-----\n");
//     }
// }
// void printint16matrix(int channels, int width, int height, int16_t* M) {
//     printf("\n");
//     for (int c = 0; c < channels; c++) {
//         for (int i = 0; i < height; i++) {
//             for (int j = 0; j < width; j++) {
//                 printf("%hu\t", M[c*height*width+i*width+j]);
//             }
//             printf("\n");
//         }
//         printf("-----\n");
//     }
// }
//
// void fill_seq_32(float* p, int n, int mode) {
//     for (int i = 0; i < n; i++) {
//         if (mode == 0) {
//             p[i] = i;
//         } else if (mode == 1) {
//             p[i] = (float)rand() / (float)(RAND_MAX);
//         } else if (mode == 2) {
//             p[i] = 1;
//         }
//     }
// }
//
//
// void fill_seq_16(int16_t* p, int n, int mode) {
//     for (int i = 0; i < n; i++) {
//         if (mode == 0) {
//             p[i] = i;
//         } else if (mode == 1) {
//           float f = (float)rand() / (float)RAND_MAX;
//           cvt_32_16(&f, &p[i], 1);
//         } else if (mode == 2) {
//             p[i] = 1;
//         }
//     }
// }
//
void setvcfg(int nd, int nw, int nh, int np) {
    int cfg = VCFG(nd, nw, nh, np);
    asm volatile ("vsetcfg %0"
                  :
                  : "r" (cfg));
}

int setvlen(int vlen) {
    int consumed;
    asm volatile ("vsetvl %0, %1"
                  : "=r" (consumed)
                  : "r" (vlen));
    asm volatile ("la t0, vsetvlen" : : : "t0");
    asm volatile ("vf 0(t0)");
    asm volatile ("fence");
    return consumed;
}
//
// void memcpy_16(int16_t* src, int16_t* dest, int len)
// {
//   setvcfg(0, 0, 1, 1);
//   for (int i = 0; i < len; ) {
//     int consumed = setvlen(len - i);
//     asm volatile ("vmca va0, %0"
//                   :
//                   : "r" (&src[i]));
//     asm volatile ("vmca va1, %0"
//                   :
//                   : "r" (&dest[i]));
//     asm volatile ("la t0, vmemcpy_16"
//                   :
//                   :
//                   : "t0");
//     asm volatile ("lw t1, 0(t0)");
//     asm volatile ("vf 0(t0)");
//     i += consumed;
//   }
//   asm volatile ("fence");
// }
//
// void memcpy_32(float* src, float* dest, int len)
// {
//   setvcfg(0, 1, 0, 1);
//   for (int i = 0; i < len; ) {
//     int consumed = setvlen(len - i);
//     asm volatile ("vmca va0, %0"
//                   :
//                   : "r" (&src[i]));
//     asm volatile ("vmca va1, %0"
//                   :
//                   : "r" (&dest[i]));
//     asm volatile ("la t0, vmemcpy_32"
//                   :
//                   :
//                   : "t0");
//     asm volatile ("lw t1, 0(t0)");
//     asm volatile ("vf 0(t0)");
//     i += consumed;
//   }
//   asm volatile ("fence");
// }
//
// void cvt_32_16(float* src, int16_t* dest, int len)
// {
//   setvcfg(0, 1, 1, 1);
//   for (int i = 0; i < len; ) {
//     int consumed = setvlen(len - i);
//     asm volatile ("vmca va0, %0"
//                   :
//                   : "r" (&src[i]));
//     asm volatile ("vmca va1, %0"
//                   :
//                   : "r" (&dest[i]));
//     asm volatile ("la t0, vcvt_32_16"
//                   :
//                   :
//                   : "t0");
//     asm volatile ("lw t1, 0(t0)");
//     asm volatile ("vf 0(t0)");
//     i += consumed;
//   }
//   asm volatile ("fence");
// }
//
// void cvt_16_32(int16_t* src, float* dest, int len)
// {
//   setvcfg(0, 1, 1, 1);
//   for (int i = 0; i < len; )
//     {
//       int consumed = setvlen(len - i);
//       asm volatile ("vmca va0, %0"
//                     :
//                     : "r" (&src[i]));
//       asm volatile ("vmca va1, %0"
//                     :
//                     : "r" (&dest[i]));
//       asm volatile ("la t0, vcvt_16_32"
//                     :
//                     :
//                     : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// }
//
//
// void gather_16(const int* id, const int16_t* src, int16_t* dest, int len) {
//     setvcfg(0, 1, 1, 2);
//     asm volatile ("la t0, vgather_16" : : : "t0");
//     asm volatile ("lw t1, 0(t0)");
//     for (int i = 0; i < len; ) {
//         int consumed = setvlen(len - i);
//
//         asm volatile ("vmcs vs1, %0" : : "r" (&src[0]));
//         asm volatile ("vmca va1, %0" : : "r" (&id[i]));
//         asm volatile ("vmca va2, %0" : : "r" (&dest[i]));
//         asm volatile ("la t0, vgather_16" : : : "t0");
//         asm volatile ("vf 0(t0)");
//         i += consumed;
//     }
//     asm volatile ("fence");
// }
//
// void gather_32(const int* id, const float* src, float* dest, int len) {
// #ifndef USE_SCALAR
//     setvcfg(0, 2, 0, 2);
//     asm volatile ("la t0, vgather_32" : : : "t0");
//     asm volatile ("lw t1, 0(t0)");
//     for (int i = 0; i < len; ) {
//         int consumed = setvlen(len - i);
//         asm volatile ("vmcs vs1, %0" : : "r" (&src[0]));
//         asm volatile ("vmca va1, %0" : : "r" (&id[i]));
//         asm volatile ("vmca va2, %0" : : "r" (&dest[i]));
//         asm volatile ("la t0, vgather_32" : : : "t0");
//         asm volatile ("vf 0(t0)");
//         i += consumed;
//     }
//     asm volatile ("fence");
// #else
//     for (int i = 0; i < len; i++)
//       dest[i] = (id[i] >= 0) ? src[id[i] >> 2] : 0.0;
// #endif
// }
//
//
// void fill_16(int N, float ALPHA, int16_t * X)
// {
//    int i;
//    setvcfg(0, 0, 1, 1);
//    asm volatile ("vmcs vs1, %0" : : "r" (ALPHA));
//    for (i = 0; i < N; )
//      {
//        int consumed = setvlen (N - i);
//        asm volatile ("vmca va0, %0" : : "r" (&X[i]));
//        asm volatile ("la t0, vfill_16" : : : "t0");
//        asm volatile ("lw t1, 0(t0)");
//        asm volatile ("vf 0(t0)");
//        i += consumed;
//      }
//    asm volatile ("fence");
// }
// void fill_32(int N, float ALPHA, float* X)
// {
// #ifndef USE_SCALAR
//   int i;
//   setvcfg(0, 0, 1, 1);
//   asm volatile ("vmcs vs1, %0" : : "r" (ALPHA));
//   for (i = 0; i < N; )
//     {
//       int consumed = setvlen (N - i);
//       asm volatile ("vmca va0, %0" : : "r" (&X[i]));
//       asm volatile ("la t0, vfill_32" : : : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// #else
//   for (int i = 0; i < N; i++)
//     X[i] = ALPHA;
// #endif
// }
//
//
// void normalize_16(int16_t *x, int16_t *mean, int16_t *variance, int filters, int spatial)
// {
//     int f, i;
//     setvcfg(0, 0, 1, 1);
//     asm volatile ("vmcs vs3, %0" : : "r" (0.000001f));
//     for(f = 0; f < filters; ++f)
//       {
//         asm volatile ("vmcs vs1, %0" : : "r" (mean[f]));
//         asm volatile ("vmcs vs2, %0" : : "r" (variance[f]));
//         for (i = 0; i < spatial ;)
//           {
//             int consumed = setvlen(spatial - i);
//             asm volatile ("vmca va0, %0" : : "r" (&x[f*spatial + i]));
//             asm volatile ("la t0, vnormalize_16" : : : "t0");
//             asm volatile ("lw t1, 0(t0)");
//             asm volatile ("vf 0(t0)");
//             i += consumed;
//           }
//     }
//     asm volatile ("fence");
// }
// void normalize_32(float *x, float *mean, float *variance, int filters, int spatial)
// {
//     int f, i;
//     setvcfg(0, 1, 0, 1);
//     asm volatile ("vmcs vs3, %0" : : "r" (0.000001f));
//     for(f = 0; f < filters; ++f)
//       {
//         asm volatile ("vmcs vs1, %0" : : "r" (mean[f]));
//         asm volatile ("vmcs vs2, %0" : : "r" (variance[f]));
//         for (i = 0; i < spatial ;)
//           {
//             int consumed = setvlen(spatial - i);
//             asm volatile ("vmca va0, %0" : : "r" (&x[f*spatial + i]));
//             asm volatile ("la t0, vnormalize_32" : : : "t0");
//             asm volatile ("lw t1, 0(t0)");
//             asm volatile ("vf 0(t0)");
//             i += consumed;
//           }
//     }
//     asm volatile ("fence");
// }
//
// void scale_16(int16_t* x, int16_t scale, int size)
// {
//   setvcfg(0, 0, 1, 1);
//   for (int i = 0; i < size; )
//     {
//       int consumed = setvlen(size - i);
//       asm volatile ("vmca va0, %0" : : "r" (&x[i]));
//       asm volatile ("vmcs vs1, %0" : : "r" (scale));
//       asm volatile ("la t0, vscale_16" : : : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// }
// void scale_32(float* x, float scale, int size)
// {
//   setvcfg(0, 1, 0, 1);
//   for (int i = 0; i < size; )
//     {
//       int consumed = setvlen(size - i);
//       asm volatile ("vmca va0, %0" : : "r" (&x[i]));
//       asm volatile ("vmcs vs1, %0" : : "r" (scale));
//       asm volatile ("la t0, vscale_32" : : : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// }
//
// void add_16(int16_t* x, int16_t y, int size)
// {
//   setvcfg(0, 0, 1, 1);
//   for (int i = 0; i < size; )
//     {
//       int consumed = setvlen(size - i);
//       asm volatile ("vmca va0, %0" : : "r" (&x[i]));
//       asm volatile ("vmcs vs1, %0" : : "r" (y));
//       asm volatile ("la t0, vadd_16" : : : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// }
// void add_32(float* x, float y, int size)
// {
// #ifndef USE_SCALAR
//   setvcfg(0, 1, 0, 1);
//   for (int i = 0; i < size; )
//     {
//       int consumed = setvlen(size - i);
//       asm volatile ("vmca va0, %0" : : "r" (&x[i]));
//       asm volatile ("vmcs vs1, %0" : : "r" (y));
//       asm volatile ("la t0, vadd_32" : : : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// #else
//   for (int i = 0; i < size; i++)
//     x[i] += y;
// #endif
// }
//
// void square_32(int N, float* X, float* dest)
// {
//   setvcfg(0, 1, 0, 1);
//   for (int i = 0; i < N; )
//     {
//       int consumed = setvlen(N - i);
//       asm volatile ("vmca va0, %0" : : "r" (&X[i]));
//       asm volatile ("vmca va1, %0" : : "r" (&dest[i]));
//       asm volatile ("la t0, vsquare_32" : : : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// }
//
// void axpy_32(int N, float A, float* X, float* Y)
// {
//   setvcfg(0, 2, 0, 1);
//   asm volatile ("vmcs vs1, %0" : : "r" (A));
//   for (int i = 0; i < N; )
//     {
//       int consumed = setvlen(N - i);
//       asm volatile ("vmca va0, %0" : : "r" (&X[i]));
//       asm volatile ("vmca va1, %0" : : "r" (&Y[i]));
//       asm volatile ("la t0, vaxpy_32" : : : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// }
//
// void mul_32(int N, float* X, float* Y)
// {
//   setvcfg(0, 2, 0, 1);
//   for (int i = 0; i < N; )
//     {
//       int consumed = setvlen(N - i);
//       asm volatile ("vmca va0, %0" : : "r" (&X[i]));
//       asm volatile ("vmca va1, %0" : : "r" (&Y[i]));
//       asm volatile ("la t0, vmul_32" : : : "t0");
//       asm volatile ("lw t1, 0(t0)");
//       asm volatile ("vf 0(t0)");
//       i += consumed;
//     }
//   asm volatile ("fence");
// }