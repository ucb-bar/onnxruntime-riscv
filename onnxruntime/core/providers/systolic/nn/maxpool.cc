// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "maxpool.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"

#ifdef SYSTOLIC_FP32

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MaxPool_nhwc,
    kOnnxDomain,
    1, 11,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MaxPool_nhwc<float>);

template <typename T>
Status MaxPool_nhwc<T>::Compute(OpKernelContext* context) const {
  fprintf(stderr, "CALLED INTO MAXPOOL\n");
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();
  const size_t input_rank = input_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_rank >= 3, "Input dimension cannot be less than 3.");
  const int64_t N = input_shape[0];
  const int64_t C = input_shape[input_rank - 1];

  ORT_ENFORCE(input_shape.Size() > 0 || N == 0, "Invalid input shape. Only N can be zero. Got:", input_shape);

  const size_t spatial_dims = input_rank - 2;
  ORT_ENFORCE(spatial_dims == 2, "We do not handle # spatial dims other than 2");
  const int64_t H = input_shape[1];
  const int64_t W = input_shape[2];

  // Compute the output size and effective padding for this pooling operation.
  std::vector<int64_t> output_dims({N});
  std::vector<int64_t> pads = pool_attrs_.pads;
  int64_t kernel_size = 1;
  int64_t input_image_size = 1;
  int64_t output_image_size = 1;
  for (size_t dim = 0; dim < spatial_dims; ++dim) {
    int64_t kernel = pool_attrs_.kernel_shape[dim];
    int64_t input_dim = input_shape[dim + 1];

    kernel_size *= kernel;
    input_image_size *= input_dim;

    int64_t output_dim = 0;
    pool_attrs_.ComputeSizePadDilations(input_dim,
                                        pool_attrs_.strides[dim],
                                        kernel,
                                        &pads.at(dim),
                                        &pads.at(spatial_dims + dim),
                                        pool_attrs_.dilations[dim],
                                        &output_dim);
    output_dims.push_back(output_dim);

    output_image_size *= output_dim;
  }
  output_dims.push_back(C);
  printf("OUTPUT DIMS %ld %ld %ld %ld\n", output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
  ORT_ENFORCE(output_dims.size() == 4, "Something went wrong. Output isn't NHWC");
  Tensor* Y = context->Output(0, output_dims);
  Tensor* I = context->Output(1, output_dims);

  const int64_t pooled_height = output_dims[1];
  const int64_t pooled_width = output_dims[2];
  const int64_t stride_h = pool_attrs_.strides[0];
  const int64_t stride_w = pool_attrs_.strides[1];
  const int64_t dilation_h = pool_attrs_.dilations[0];
  const int64_t dilation_w = pool_attrs_.dilations[1];

  std::vector<int64_t> kernel_shape = pool_attrs_.kernel_shape;

  const auto* X_data = X->template Data<T>();
  auto* Y_data = Y->template MutableData<T>();
  int64_t* I_data = I != nullptr ? I->template MutableData<int64_t>() : nullptr;

  for (int batch = 0; batch < N; batch++) {
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        for (int c = 0; c < C; c++) {
          const int64_t pool_index = batch * pooled_width * pooled_height * C + (ph * pooled_width + pw) * C + c;
          T Yh = std::numeric_limits<T>::lowest();
          int64_t h_index = -1;
          int64_t w_index = -1;
          for (int64_t h = hstart; h < hend; h += dilation_h) {
            if (math::is_a_ge_zero_and_a_lt_b(h, H)) {
              for (int64_t w = wstart; w < wend; w += dilation_w) {
                if (math::is_a_ge_zero_and_a_lt_b(w, W)) {
                  const int64_t input_index = batch * H * W * C + (h * W + w) * C + c;
                  if (X_data[input_index] > Yh) {
                    Yh = X_data[input_index];
                    h_index = h;
                    w_index = w;
                  }
                }
              }
            }
          }
          Y_data[pool_index] = Yh;
          if (I_data != nullptr)
            I_data[pool_index] = batch * H * W * C + (h_index * W + w_index) * C + c;
        }
      }
    }
  }
  // if (I) {
  //   printf("Index tensor\n");
  //   DumpTensor<int64_t>(I);
  // }

  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime

#endif