#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace systolic {


template <typename T, StorageOrder kOrder>
void ComputeMaxPool2D(
    int W,
    int t,
    int b,
    int l,
    int r,
    int y,
    const ConstEigenArrayMap<T>& X_arr,
    EigenArrayMap<T>* Y_arr);

template <>
void ComputeMaxPool2D<int8_t, StorageOrder::NHWC>(
    const int W,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<int8_t>& X_arr,
    EigenArrayMap<int8_t>* Y_arr) {
  Y_arr->col(y).setConstant(std::numeric_limits<int8_t>::lowest());
  for (int i = t; i < b; ++i) {
    for (int j = l; j < r; ++j) {
      Y_arr->col(y) = Y_arr->col(y).max(X_arr.col(i * W + j));
    }
  }
}

template <typename T, StorageOrder kOrder>
void RunMaxPool2D(
    const int N,
    const int C,
    const int X_H,
    const int X_W,
    const int Y_H,
    const int Y_W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const T* X,
    T* Y) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  const T* X_ptr = X;
  T* Y_ptr = Y;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
                                      ? ConstEigenArrayMap<T>(X_ptr, X_W, X_H)
                                      : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
    EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
                                 ? EigenArrayMap<T>(Y_ptr, Y_W, Y_H)
                                 : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
    for (int h = 0; h < Y_H; ++h) {
      const int t = std::max(h * stride_h - pad_t, 0);
      const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
      for (int w = 0; w < Y_W; ++w) {
        const int l = std::max(w * stride_w - pad_l, 0);
        const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
        const int y = h * Y_W + w;
        ComputeMaxPool2D<T, kOrder>(X_W, t, b, l, r, y, X_arr, &Y_arr);
      }
    }
    X_ptr += X_stride;
    Y_ptr += Y_stride;
  }
}

}  // namespace systolic
}  // namespace onnxruntime
