/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "qlinearconvtranspose.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"
#include "conv_pool_helper.h"

#ifdef SYSTOLIC_INT8

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    QLinearConvTranspose,
    kOnnxDomain,
    1, 11,
    int8_t,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConvTranspose<int8_t>);

template <typename T>
Status QLinearConvTranspose<T>::Compute(OpKernelContext* context) const {
  return QLinearConvTranspose<T>::DoConvTranspose(context);
}

template <typename T>
Status QLinearConvTranspose<T>::DoConvTranspose(OpKernelContext* context) const {

  // This is currently stubbed out because we can't properly handle bias addition
  // The bias addition (where bias is int32) needs to be done after the col2im,
  // so we can't do this on gemmini. We also can't do this on CPU 
  // because the output scaling factor given to us is computed /after/ the bias was added.
  // So we would either have to modify the quantization to compute two scaling factors:
  // one before bias addition and one after, so we could do the bias add on CPU.
  // But that's too hacky since the same conv transpose implementation used for convgrad
  // Can apply here as well, and with that we don't need the col2im.
  // Of course, with that we would need to implement it in NHWC.
  // That's left as an exercise to the reader (follow the same we we do for qlinearconv).
  ORT_ENFORCE(false, "QLinearConvTranspose unimplemented. See the comment");
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  QLinearConvTransposeAttributes::Prepare p;
  bool has_bias = num_inputs == 9;
  ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p));

  // Bail out early if one of the dimensions is zero.
  if (p.Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = p.input_shape.Size();
  const int64_t X_offset = p.num_input_channels / conv_transpose_attrs_.group * input_image_size;
  const int64_t Y_offset = p.Y->Shape().Size() / p.Y->Shape()[0] / conv_transpose_attrs_.group;
  const int64_t W_offset = p.F->Shape().Size() / conv_transpose_attrs_.group;
  const int64_t kernel_size = TensorShape(p.kernel_shape).Size();
  const int64_t kernel_dim = p.num_output_channels / conv_transpose_attrs_.group * kernel_size;
//  const int64_t output_size = (p.Y->Shape().Slice(2)).Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const int64_t col_buffer_size = kernel_dim * p.input_shape.Size();
  auto col_data = alloc->Alloc(SafeInt<size_t>(sizeof(T)) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  const T* Xdata = p.X->template Data<T>();
  const T* filter_data = p.F->template Data<T>();
  T* Ydata = p.Y->template MutableData<T>();
  ORT_UNUSED_PARAMETER(W_offset);
  ORT_UNUSED_PARAMETER(filter_data);

  ORT_ENFORCE(p.X->Shape().NumDimensions() == 4, "Cannot handle anything other than 2D Conv Transpose atm");

  char acc_mode = static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode();
  ORT_UNUSED_PARAMETER(acc_mode);

  for (auto image_id = 0; image_id < p.N; ++image_id) {
    for (int group_id = 0; group_id < conv_transpose_attrs_.group; ++group_id) {
      printf("In qlinearconv transpose\n");
      // This is commented out because mlas hasn't been updated to expose SystolicGemm in int8 mode
      // Weight term
    //   SystolicGemm(
    //       acc_mode,
    //       true, /*transA */
    //       false, /*transB */
    //       kernel_dim,
    //       input_image_size,
    //       p.num_input_channels / conv_transpose_attrs_.group,
    //       1,
    //       filter_data + group_id * W_offset,
    //       Xdata + group_id * X_offset,
    //       0,
    //       col_buffer_data,
    //       1.0f /* scaleFactor */);

      // Col2im
      Col2imNCHW(
          col_buffer_data,
          p.num_output_channels / conv_transpose_attrs_.group,
          p.Y->Shape()[2],
          p.Y->Shape()[3],
          p.kernel_shape[0],
          p.kernel_shape[1],
          p.dilations[0],
          p.dilations[1],
          p.pads[0],
          p.pads[1],
          p.pads[2],
          p.pads[3],
          p.strides[0],
          p.strides[1],
          Ydata + group_id * Y_offset);
    }

    // Commented out because we can't handle bias
    // if (p.B != nullptr) {
    //   auto Ymatrix = EigenMatrixMap<T>(Ydata, output_size, p.num_output_channels);
    //   auto Bvec = ConstEigenVectorMap<T>(p.B->template Data<T>(), p.num_output_channels);
    //   Ymatrix.rowwise() += Bvec.transpose();
    // }

    Xdata += X_offset * conv_transpose_attrs_.group;
    Ydata += Y_offset * conv_transpose_attrs_.group;
  }


  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime

#endif