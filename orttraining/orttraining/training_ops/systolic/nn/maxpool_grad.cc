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

#include "maxpool_grad.h"
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

template <typename T>
Status MaxPoolGrad_nhwc<T>::Compute(OpKernelContext* context) const {
  // This implementation is identical to the NCHW version, we copy it
  // just in case that for some reason they change the implementation of the NCHW later
  // to actually make use of the NCHW-ness
  std::vector<VectorInt64> output_tensor_shapes = {};

  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* indices = context->Input<Tensor>(1);
  ORT_ENFORCE(dY->Shape() == indices->Shape(), "The shape of dY and indices does not match in MaxPoolGrad.");

  TensorShape dX_shape(output_tensor_shapes_[0]);
  dX_shape[0] = dY->Shape()[0];
  Tensor* dX = context->Output(0, dX_shape);

  const T* dY_data = dY->template Data<T>();
  const int64_t* indices_data = indices->template Data<int64_t>();
  T* dX_data = dX->template MutableData<T>();

  EigenVectorMap<T>(dX_data, dX_shape.Size()).setZero();

  for (int64_t i = 0; i < dY->Shape().Size(); ++i) {
    T* p_dX_data = dX_data + indices_data[i];
    *p_dX_data += dY_data[i];
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MaxPoolGrad_nhwc,
    kOnnxDomain,
    9,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MaxPoolGrad_nhwc<float>);

}  // namespace systolic
}  // namespace onnxruntime

#endif