// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_add.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

#ifdef SYSTOLIC_INT8

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearAdd,
    kMSDomain,
    1,
    int8_t,
    kSystolicExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    QLinearAdd<int8_t>);

namespace {
struct QLinearBroadcastHelper : public BroadcastHelper {
  QLinearBroadcastHelper(InputBroadcaster& input_broadcaster,
                         OutputBroadcaster& output_broadcaster,
                         ThreadPool* threadpool,
                         double unit_cost,
                         float A_scale_in, float B_scale_in, float C_scale_in, bool relu,
                         char accelerator_mode)
      : BroadcastHelper{input_broadcaster, output_broadcaster, nullptr, threadpool, unit_cost},
        A_scale{A_scale_in},
        B_scale{B_scale_in},
        C_scale{C_scale_in},
        relu{relu},
        accelerator_mode{accelerator_mode} {
  }

  QLinearBroadcastHelper(const QLinearBroadcastHelper& rhs, size_t offset, size_t num_elements)
      : BroadcastHelper(rhs, offset, num_elements),
        A_scale{rhs.A_scale},
        B_scale{rhs.B_scale},
        C_scale{rhs.C_scale},
        relu{rhs.relu},
        accelerator_mode{rhs.accelerator_mode} {
  }

  float A_scale;
  float B_scale;
  float C_scale;
  uint8_t A_zero_point{0};
  uint8_t B_zero_point{0};
  uint8_t C_zero_point{0};
  bool relu;
  char accelerator_mode;
};

template <typename T>
void QLinearImpl(OpKernelContext& context, double unit_cost, const ProcessBroadcastSpanFuncs& functors, bool relu, char acc_mode) {
  auto tensor_a_scale = context.Input<Tensor>(1);
  auto tensor_a_zero_point = context.Input<Tensor>(2);
  auto tensor_b_scale = context.Input<Tensor>(4);
  auto tensor_b_zero_point = context.Input<Tensor>(5);
  auto tensor_c_scale = context.Input<Tensor>(6);
  auto tensor_c_zero_point = context.Input<Tensor>(7);

  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_a_scale),
              "QLinearAdd : input1 A_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_a_zero_point == nullptr || IsScalarOr1ElementVector(tensor_a_zero_point),
              "QLinearAdd : input1 A_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_b_scale),
              "QLinearAdd : input1 B_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_b_zero_point == nullptr || IsScalarOr1ElementVector(tensor_b_zero_point),
              "QLinearAdd : input1 B_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_c_scale),
              "QLinearAdd : input1 C_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_c_zero_point == nullptr || IsScalarOr1ElementVector(tensor_c_zero_point),
              "QLinearAdd : input1 C_zero_point must be a scalar or 1D tensor of size 1 if given");

  const float A_scale = *(tensor_a_scale->Data<float>());
  const T A_zero_point = (nullptr == tensor_a_zero_point) ? T{} : *(tensor_a_zero_point->template Data<T>());
  ORT_ENFORCE(A_zero_point == 0, "Systolic can only handle zero-point of zero");
  const float B_scale = *(tensor_b_scale->Data<float>());
  const T B_zero_point = (nullptr == tensor_b_zero_point) ? T{} : *(tensor_b_zero_point->template Data<T>());
  ORT_ENFORCE(B_zero_point == 0, "Systolic can only handle zero-point of zero");
  const float C_scale = *(tensor_c_scale->Data<float>());
  const T C_zero_point = (nullptr == tensor_c_zero_point) ? T{} : *(tensor_c_zero_point->template Data<T>());
  ORT_ENFORCE(C_zero_point == 0, "Systolic can only handle zero-point of zero");

  InputBroadcaster input_broadcaster{*context.Input<Tensor>(0), *context.Input<Tensor>(3)};
  OutputBroadcaster output_broadcaster{input_broadcaster.GetSpanSize(),
                                       *context.Output(0, input_broadcaster.GetOutputShape())};

  QLinearBroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster,
                                          context.GetOperatorThreadPool(), unit_cost,
                                          A_scale, B_scale, C_scale, relu, acc_mode);

  BroadcastLooper(broadcast_helper, functors);
}
}  // namespace

template <typename T>
Status QLinearAdd<T>::Compute(OpKernelContext* context) const {
  const ProcessBroadcastSpanFuncs functors = {
      [](BroadcastHelper& per_iter_bh) {
        // We don't yet support scalar + matrix resadd on systolic
        // We could do this via SW only by manually broadcasting
        // to systolic size and then mvin with 0 stride
        ORT_UNUSED_PARAMETER(per_iter_bh);
        ORT_NOT_IMPLEMENTED("Scalar + Matrix resadd on systolic not implemented");
        // QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        // const T input0 = per_iter_bh.ScalarInput0<T>();
        // auto input1 = per_iter_bh.SpanInput1<T>();
        // auto output = per_iter_bh.OutputSpan<T>();

        // SystolicAdd(
        //     qlbh.accelerator_mode,
        //     qlbh.relu,
        //     input0, qlbh.A_scale,
        //     input1.data(), qlbh.B_scale,
        //     output.data(), qlbh.C_scale,
        //     output.size());
      },
      [](BroadcastHelper& per_iter_bh) {
        ORT_UNUSED_PARAMETER(per_iter_bh);
        ORT_UNUSED_PARAMETER(per_iter_bh);
        // QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        // auto input0 = per_iter_bh.SpanInput0<T>();
        // const T input1 = per_iter_bh.ScalarInput1<T>();
        // auto output = per_iter_bh.OutputSpan<T>();
        // SystolicAdd(
        //     qlbh.accelerator_mode,
        //     qlbh.relu,
        //     input0.data(), qlbh.A_scale,
        //     input1, qlbh.B_scale,
        //     output.data(), qlbh.C_scale,
        //     output.size());
      },
      [](BroadcastHelper& per_iter_bh) {
        QLinearBroadcastHelper& qlbh = static_cast<QLinearBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        SystolicAdd(
            qlbh.accelerator_mode,
            qlbh.relu,
            input0.data(), qlbh.A_scale,
            input1.data(), qlbh.B_scale,
            output.data(), qlbh.C_scale,
            output.size());
      }};

  QLinearImpl<T>(*context, 1.0, functors, this->fused_relu_,
    static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode());

  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime

#endif