// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/providers/cpu/math/gemm_helper.h"

#ifdef SYSTOLIC_FP32

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7, 13,
    float,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

template <typename T>
static void GemmBroadcastBias(int64_t M, int64_t N, float beta,
                              const T* c_data, const TensorShape* c_shape,
                              T* y_data) {
  // Broadcast the bias as needed if bias is given
  if (beta != 0 && c_data != nullptr) {
    ORT_ENFORCE(c_shape != nullptr, "c_shape is required if c_data is provided");
    auto output_mat = EigenMatrixMapRowMajor<T>(y_data, M, N);
    if (c_shape->Size() == 1) {
      // C is (), (1,) or (1, 1), set the scalar
      output_mat.setConstant(*c_data);
    } else if (c_shape->NumDimensions() == 1 || (*c_shape)[0] == 1) {
      // C is (N,) or (1, N)
      output_mat.rowwise() = ConstEigenVectorMap<T>(c_data, N).transpose();
    } else if ((*c_shape)[1] == 1) {
      // C is (M, 1)
      output_mat.colwise() = ConstEigenVectorMap<T>(c_data, M);
    } else {
      // C is (M, N), no broadcast needed.
      output_mat = ConstEigenMatrixMapRowMajor<T>(c_data, M, N);
    }
  }
}

template <>
Status Gemm<float>::Compute(OpKernelContext* context) const {
  const auto* A = context->Input<Tensor>(0);
  const auto* B = context->Input<Tensor>(1);
  const auto* C = context->Input<Tensor>(2);

  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(A->Shape(), trans_A_, B->Shape(), trans_B_,
                    C != nullptr ? C->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  int64_t M = helper.M();
  int64_t N = helper.N();
  int64_t K = helper.K();

  //printf("M, N, K: %d %d %d\n", (int) M, (int) N, (int) K);

  auto Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  float* y_data = Y->MutableData<float>();

  const float* c_data = C != nullptr ? C->Data<float>() : nullptr;
  const TensorShape* c_shape = C != nullptr ? &C->Shape() : nullptr;

  // if input is empty tensor, return directly as nothing need to be calculated.
  if (M == 0 || N == 0)
    return Status::OK();

  // Broadcast the bias as needed if bias is given
  GemmBroadcastBias(M, N, beta_, c_data, c_shape, y_data);

  // printf("A matrix\n");
  // PrintMatrix(M, K, A->Data<float>());
  // printf("B matrix\n");
  // PrintMatrix(K, N, B->Data<float>());
  // printf("Bias matrix\n");
  // PrintMatrix(M, N, y_data);

  char acc_mode = static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode();
  SystolicGemm(acc_mode,
                trans_A_, trans_B_, M, N, K,
                alpha_, A->Data<float>(), B->Data<float>(), c_data != nullptr ? beta_ : 0, y_data);

  // printf("Out matrix\n");
  // PrintMatrix(M, N, y_data);

  // printf("First few output values\n:");
  // for (int i = 0; i < 20; i++) {
  //   printf("%f ", y_data[i]);
  // }
  // printf("\n");

  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime

#endif