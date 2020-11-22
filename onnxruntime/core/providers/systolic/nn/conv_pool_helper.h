#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"
#include "core/common/exceptions.h"
#include "core/providers/common.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
void Im2Col_NHWC(const T* data_im, int64_t channels, int64_t height,
                                               int64_t width, int64_t kernel_h, int64_t kernel_w,
                                               int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
                                               int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
                                               int64_t stride_w, T* data_col, const int64_t groups, T padding_value) {
  /* https://github.com/pytorch/pytorch/blob/master/caffe2/quantization/server/im2col_dnnlowp.h#L186 */
  /* https://github.com/pytorch/pytorch/blob/master/caffe2/utils/math_cpu.cc#L2383 */

  const int64_t dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int64_t height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int64_t width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  for (int h = 0; h < height_col; ++h) {
    int h_pad = -pad_t + h * stride_h;
    T* data_col_temp =
        data_col + h * width_col * kernel_h * kernel_w * channels;
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      int r = 0;
      for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
        int s = 0;
        for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w, ++s) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            for (int g = 0; g < groups; ++g) {
              memcpy(
                  data_col_temp +
                      ((g * kernel_h + r) * kernel_w + s) * (channels / groups),
                  data_im + (ih * width + iw) * channels +
                      g * (channels / groups),
                  sizeof(T) * (channels / groups));
            }
          } else {
            // This should be simply padded with zero.
            for (int g = 0; g < groups; ++g) {
              for (int i = 0; i < channels / groups; ++i) {
                data_col_temp
                    [(((g * kernel_h + r) * kernel_w) + s) *
                         (channels / groups) +
                     i] = padding_value;
              }
            }
          }
        }  // for each iw
      }    // for each ih
      data_col_temp += kernel_h * kernel_w * channels;
      w_pad += stride_w;
    }  // for each output pixel
  }    // for each image row
}

Status ValidateConvInputShapeNHWC(const Tensor* X, const Tensor* W, const int64_t group) {
  const int64_t C = X->Shape()[3];
  const int64_t M = W->Shape()[3];
  const int64_t C_over_groups = W->Shape()[2];

  if (X->Shape().NumDimensions() != W->Shape().NumDimensions()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "X num_dims does not match W num_dims.",
                            " X: ", X->Shape().ToString().c_str(),
                            " W: ", W->Shape().ToString().c_str());
  }

  if (C != C_over_groups * group) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input channels C is not equal to kernel channels * group.",
                            " C: ", C,
                            " kernel channels: ", C_over_groups,
                            " group: ", group);
  }

  if (M % group != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output channels M is not divisible by group.",
                            " M: ", M,
                            " group: ", group);
  }
  return Status::OK();
}

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
