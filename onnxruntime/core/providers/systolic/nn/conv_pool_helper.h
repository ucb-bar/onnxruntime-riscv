#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/exceptions.h"
#include "core/providers/common.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace systolic {

/**
 * Try to run a given NHWC conv on systolic if possible
 * Gemmini only supports groups == 1. 
 * Additionally, we must have strides and padding must be equal
 * Kernel dimensions must also be square (W == H)
 * 
 * Note that all inputs/outputs/weights are here in NHWC format
 * Bias is null if no bias
 * @return true If successfully ran on systolic
 */

template<typename elem_t, typename bias_t>
inline bool TryConvOnSystolic(char accelerator_mode,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& strides,
                              int64_t groups,
                              const Tensor* X,
                              const Tensor* W,
                              const Tensor* B,
                              Tensor* output,
                              const TensorShape& Y_dims_prepool,
                              const TensorShape& Y_dims_postpool,
                              bool relu,
                              const PoolAttributes *pool_attrs_,
                              float output_scale) {
  if (groups != 1) {
    return false;
  }

  int input_dim, output_dim, kernel_dim;

  // If input H != W
  if ((input_dim = X->Shape()[1]) != X->Shape()[2]) {
    return false;
  }

  // If output H != W
  // N.B., systolic takes as input the dimension before pooling
  if ((output_dim = Y_dims_prepool[1]) != Y_dims_prepool[2]) {
    return false;
  }

  // If Kernel kH != hW
  if ((kernel_dim = W->Shape()[0]) != W->Shape()[1]) {
    return false;
  }

  // All dilations must be equal to 1.
  if (std::any_of(dilations.begin(), dilations.end(), [&](int i) { return i != 1; })) {
    return false;
  }

  // All pads must be the same
  if (std::any_of(pads.begin(), pads.end(), [&](int i) { return i != pads[0]; })) {
    return false;
  }

  // All strides must be the same
  if (std::any_of(strides.begin(), strides.end(), [&](int i) { return i != strides[0]; })) {
    return false;
  }

  int pool_size = 0;
  int pool_stride = 0;
  int pool_padding = 0;
  if (pool_attrs_ && pool_attrs_->fused_pool) {
    // printf("Checking pool attrs %zd %zd %zd %zd %zd \n", pool_attrs_.kernel_shape[0], pool_attrs_.kernel_shape[1],
    //     pool_attrs_.strides[0],  pool_attrs_.strides[1]),
    //      pool_attrs_.pads[0], pool_attrs_.pads[1]);
    if ((pool_size = pool_attrs_->kernel_shape[0]) != pool_attrs_->kernel_shape[1]) {
      return false;
    }
    if ((pool_stride = pool_attrs_->strides[0]) != pool_attrs_->strides[1]) {
      return false;
    }
    if ((pool_padding = pool_attrs_->pads[0]) != pool_attrs_->pads[1]) {
      return false;
    }
  }

  const auto* Xdata = X->template Data<elem_t>();
  const auto* Wdata = W->template Data<elem_t>();
  const auto* Bdata = B != nullptr ? B->template Data<bias_t>() : nullptr;

  // Assume that tensor allocated to us is already sized appropriately.
  // That is, if pooling is to be applied the size is post-pool.
  Tensor* Y = output;

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return true;
  }

  auto* Ydata = Y->template MutableData<elem_t>();

  int batch_size = X->Shape()[0];
  int input_channels = X->Shape()[3];
  int output_channels = W->Shape()[3];

  SystolicConv(accelerator_mode,
               batch_size,
               input_dim,
               input_channels,
               output_channels,
               output_dim,
               strides[0],
               pads[0],
               kernel_dim,
               /*input= */ Xdata,
               /*weights= */ Wdata,
               /*bias= */ Bdata,
               /*output= */ Ydata,
               /*relu =  */ relu,
               /* output_scale= */ output_scale,
               pool_size,
               pool_stride,
               pool_padding);

  //printf("First few output data %d %d %d %d\n", Ydata[0], Ydata[1], Ydata[2], Ydata[3]);
  return true;
}


template<typename elem_t, typename bias_t>
inline bool TryConvTransposeOnSystolic(char accelerator_mode,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& strides,
                              int64_t groups,
                              const Tensor* input,
                              const Tensor* W,
                              const Tensor* B,
                              Tensor* output,
                              bool relu,
                              float output_scale) {
  if (groups != 1) {
    return false;
  }

  int input_dim, output_dim, kernel_dim;

  // If input H != W
  if ((input_dim = input->Shape()[1]) != input->Shape()[2]) {
    return false;
  }

  // If output H != W
  if ((output_dim = output->Shape()[1]) !=  output->Shape()[2]) {
    return false;
  }

  // If Kernel kH != hW
  if ((kernel_dim = W->Shape()[0]) != W->Shape()[1]) {
    return false;
  }

  // All dilations must be equal to 1.
  if (std::any_of(dilations.begin(), dilations.end(), [&](int i) { return i != 1; })) {
    return false;
  }

  // All pads must be the same
  if (std::any_of(pads.begin(), pads.end(), [&](int i) { return i != pads[0]; })) {
    return false;
  }

  // All strides must be the same
  if (std::any_of(strides.begin(), strides.end(), [&](int i) { return i != strides[0]; })) {
    return false;
  }

  const auto* input_data = input->template Data<elem_t>();
  const auto* Wdata = W->template Data<elem_t>();
  const auto* Bdata = B != nullptr ? B->template Data<bias_t>() : nullptr;

  // Bail out early if one of the dimensions is zero.
  if (output->Shape().Size() == 0) {
    return true;
  }

  auto* output_data = output->template MutableData<elem_t>();

  int batch_size = input->Shape()[0];
  int input_channels = input->Shape()[3];
  int output_channels = output->Shape()[3];

  printf("Calling into systolicConvTranspose(/*batch_size = */  %d, /*input_dim = */ %d,"
                      "/* input_channels = */ %d , /*output_channels = */ %d, /*output_dim=*/ %d, /*stride=*/%d,"
                      "/*pad=*/ %d, /*kernel_dim=*/ %d",
               batch_size,
               input_dim,
               input_channels,
               output_channels,
               output_dim,
               (int) strides[0],
               (int) pads[0],
               kernel_dim);

  SystolicConvTranspose(accelerator_mode,
               batch_size,
               input_dim,
               input_channels,
               output_channels,
               output_dim,
               strides[0],
               pads[0],
               kernel_dim,
               /*input= */ input_data,
               /*weights= */ Wdata,
               /*bias= */ Bdata,
               /*output= */ output_data,
               /*relu =  */ relu,
               /* output_scale= */ output_scale);

  printf("First few output data %f %f %f %f\n", output_data[0], output_data[1], output_data[2], output_data[3]);
  return true;
}


template<typename elem_t, typename bias_t>
inline bool TryConvBackpropFilterOnSystolic(char accelerator_mode,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& strides,
                              int64_t groups,
                              const Tensor* input,
                              const Tensor* W,
                              const Tensor* B,
                              Tensor* output,
                              bool relu,
                              float output_scale) {
  if (groups != 1) {
    return false;
  }

  int input_dim, output_dim, kernel_dim;

  // If input H != W. Note that H, W are fixed by NHWC_to_CHWN
  if ((input_dim = input->Shape()[1]) != input->Shape()[2]) {
    return false;
  }

  // Note that output shape given to us in HWIO form
  if ((output_dim = output->Shape()[0]) !=  output->Shape()[1]) {
    return false;
  }

  // Kernel will undergo NHWC_to_HWNC transform
  if ((kernel_dim = W->Shape()[1]) != W->Shape()[2]) {
    return false;
  }

  // All dilations must be equal to 1.
  if (std::any_of(dilations.begin(), dilations.end(), [&](int i) { return i != 1; })) {
    return false;
  }

  // All pads must be the same
  if (std::any_of(pads.begin(), pads.end(), [&](int i) { return i != pads[0]; })) {
    return false;
  }

  // All strides must be the same
  if (std::any_of(strides.begin(), strides.end(), [&](int i) { return i != strides[0]; })) {
    return false;
  }

  const auto* input_data = input->template Data<elem_t>();
  const auto* Wdata = W->template Data<elem_t>();
  const auto* Bdata = B != nullptr ? B->template Data<bias_t>() : nullptr;

  // Bail out early if one of the dimensions is zero.
  if (output->Shape().Size() == 0) {
    return true;
  }

  auto* output_data = output->template MutableData<elem_t>();

  // N.b. the batch size is taken after the NHWC_to_CHWN transform
  int batch_size = input->Shape()[3];
  // N.b. the input channels are after the NHWC_to_CHWN
  int input_channels = input->Shape()[0];
  int output_channels = output->Shape()[3];

  printf("Calling into SystolicConvBackpropFilter(/*batch_size = */ %d, /*input_dim = */ %d,"
                      "/* input_channels = */ %d , /*output_channels = */ %d, /*output_dim=*/ %d, /*stride=*/%d,"
                      "/*pad=*/ %d, /*kernel_dim=*/ %d",
               batch_size,
               input_dim,
               input_channels,
               output_channels,
               output_dim,
               (int) strides[0],
               (int) pads[0],
               kernel_dim);

  SystolicConvBackpropFilter(accelerator_mode,
               batch_size,
               input_dim,
               input_channels,
               output_channels,
               output_dim,
               strides[0],
               pads[0],
               kernel_dim,
               /*input= */ input_data,
               /*weights= */ Wdata,
               /*bias= */ Bdata,
               /*output= */ output_data,
               /*relu =  */ relu,
               /* output_scale= */ output_scale);

  printf("First few output data %f %f %f %f\n", output_data[0], output_data[1], output_data[2], output_data[3]);
  return true;
}

template<typename T>
void EigenAdd(int N, const T* a, const T* b, T* y) {
  EigenVectorMap<T>(y, N) = ConstEigenVectorMap<T>(a, N).array() + ConstEigenVectorMap<T>(b, N).array();
}

/* https://github.com/pytorch/pytorch/blob/master/caffe2/utils/math_cpu.cc#L2383 */
inline void Col2Im_NHWC(const float* data_col, int64_t channels, int64_t height,
                                                    int64_t width, int64_t kernel_h, int64_t kernel_w,
                                                    int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
                                                    int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
                                                    int64_t stride_w, float* data_im, int groups) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  memset(data_im, 0, height * width * channels * sizeof(float));  
  const int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  int h_pad = -pad_t;

  if (groups == 1) {
    for (int h = 0; h < height_col; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < width_col; ++w) {
        for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
          if (!(ih >= 0 && ih < height)) {
            data_col += kernel_w * channels;
            continue;
          }
          for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
            if (iw >= 0 && iw < width){
              float* img_data_patch = data_im + (ih * width + iw) * channels;
              EigenAdd(channels, img_data_patch, data_col, img_data_patch);
            }
            data_col += channels;
          } // iw
        } // ih
        w_pad += stride_w;
      } // w
      h_pad += stride_h;
    } // h
  } else {
    const int C_per_G = channels / groups;
    for (int h = 0; h < height_col; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < width_col; ++w) {
        int r = 0;
        for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
          int s = 0;
          for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w, ++s) {
            if ((ih >= 0 && ih < height) &&
                (iw >= 0 && iw < width)) {
              float* img_data_patch = data_im + (ih * width + iw) * channels;
              for (int g = 0; g < groups; ++g) {
                EigenAdd(
                    C_per_G,
                    img_data_patch + g * C_per_G,
                    data_col + ((g * kernel_h + r) * kernel_w + s) * C_per_G,
                    img_data_patch + g * C_per_G);
              }
            }
          } // iw
        } // ih
        data_col += kernel_h * kernel_w * channels;
        w_pad += stride_w;
      } // w
      h_pad += stride_h;
    } // h
  }
}

template <typename T>
inline void Im2Col_NHWC(const T* data_im, int64_t channels, int64_t height,
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

inline Status ValidateConvInputShapeNHWC(const Tensor* X, const Tensor* W, const int64_t group) {
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
inline void ComputeMaxPool2D(
    int W,
    int t,
    int b,
    int l,
    int r,
    int y,
    const ConstEigenArrayMap<T>& X_arr,
    EigenArrayMap<T>* Y_arr);

// https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
template <>
inline void ComputeMaxPool2D<int8_t, StorageOrder::NHWC>(
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
inline void RunMaxPool2D(
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

template<typename T>
void Col2imNCHW(const T* data_col, int64_t channels, int64_t height,
                                                    int64_t width, int64_t kernel_h, int64_t kernel_w,
                                                    int64_t dilation_h, int64_t dilation_w, int64_t pad_t,
                                                    int64_t pad_l, int64_t pad_b, int64_t pad_r, int64_t stride_h,
                                                    int64_t stride_w, T* data_im) {
  const int64_t output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int64_t output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;
  const int64_t hwc = height * width * channels;
  memset(data_im, 0, gsl::narrow<ptrdiff_t>(hwc) * sizeof(T));   

  // Fast path for zero padding and no dilation
  // From Torch, modified THNN_(unfolded_acc)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
      pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      const auto* dst = data_col +
                        nip * (kernel_h * kernel_w * output_h * output_w) +
                        kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
      auto* src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          auto offsrc = src + (iy * width + ix);
          const auto offdst = dst + (y * output_w);
          for (auto i = 0; i < output_w; ++i) {
            offsrc[i] += offdst[i];
          }
        } else {
          for (auto x = 0; x < output_w; x++) {
            auto offsrc = src + (iy * width + ix + x * stride_w);
            const auto offdst = dst + (y * output_w + x);
            *offsrc += *offdst;
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    // From Intel, https://github.com/BVLC/caffe/pull/3536
    const int64_t pad_h = pad_t;
    const int64_t pad_w = pad_l;
    const int64_t channel_size = height * width;
    for (int64_t channel = channels; channel--; data_im += channel_size) {
      for (int64_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int64_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int64_t input_row = -pad_h + kernel_row * dilation_h;
          for (int64_t output_rows = output_h; output_rows; output_rows--) {
            if (!(input_row >= 0 && input_row < height)) {
              data_col += output_w;
            } else {
              int64_t input_col = -pad_w + kernel_col * dilation_w;
              for (int64_t output_col = output_w; output_col; output_col--) {
                if ((input_col >= 0 && input_col < width)) {
                  data_im[input_row * width + input_col] += *data_col;
                }
                data_col++;
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
    return;
  }

  // Fallback
  const int64_t dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int64_t height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int64_t width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  int64_t channels_col = channels * kernel_h * kernel_w;
  for (int64_t c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % kernel_w;
    int64_t h_offset = (c / kernel_w) % kernel_h;
    int64_t c_im = c / kernel_h / kernel_w;
    for (int64_t h = 0; h < height_col; ++h) {
      for (int64_t w = 0; w < width_col; ++w) {
        int64_t h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int64_t w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
        }
      }
    }
  }
}

}  // namespace systolic
}  // namespace onnxruntime
