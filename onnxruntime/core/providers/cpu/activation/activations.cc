// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"

namespace onnxruntime {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(alias, x, sinceVersion)                              \
  ONNX_CPU_OPERATOR_KERNEL(                                                                          \
      alias,                                                                                         \
      sinceVersion,                                                                                  \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      x<float>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion) \
  REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(x, x, sinceVersion)

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid, 6);
// SoftPlus is the default case for ParametricSoftPlus
REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(Softplus, ParametricSoftplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 10);

const struct {
    float LowerRange;
    float UpperRange;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_10;
    float beta_8;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
    float one_half;
} MlasLogisticConstants = {
    -18.0f,
    18.0f,
    4.37031012579801e-11f,
    1.15627324459942e-07f,
    6.08574864600143e-05f,
    8.51377133304701e-03f,
    2.48287947061529e-01f,
    6.10247389755681e-13f,
    5.76102136993427e-09f,
    6.29106785017040e-06f,
    1.70198817374094e-03f,
    1.16817656904453e-01f,
    9.93151921023180e-01f,
    0.5f,
};

template <>
Status Sigmoid<float>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  Tensor* Y = context->Output(0, x_shape);


  const float *input = X->template Data<float>();
  float *output = Y->template MutableData<float>();
  size_t numElem = x_shape.Size();

  for (size_t i = 0; i < numElem; i++) {
      float Value = (std::min)(MlasLogisticConstants.UpperRange, (std::max)(MlasLogisticConstants.LowerRange, input[i]));

      float ValueSquared = Value * Value;

      float p;
      p = ValueSquared * MlasLogisticConstants.alpha_9 + MlasLogisticConstants.alpha_7;
      p = p * ValueSquared + MlasLogisticConstants.alpha_5;
      p = p * ValueSquared + MlasLogisticConstants.alpha_3;
      p = p * ValueSquared + MlasLogisticConstants.alpha_1;
      p = p * Value;

      float q;
      q = ValueSquared * MlasLogisticConstants.beta_10 + MlasLogisticConstants.beta_8;
      q = q * ValueSquared + MlasLogisticConstants.beta_6;
      q = q * ValueSquared + MlasLogisticConstants.beta_4;
      q = q * ValueSquared + MlasLogisticConstants.beta_2;
      q = q * ValueSquared + MlasLogisticConstants.beta_0;

      output[i] = (p / q) + 0.5f;
  }

  return Status::OK();
}

const struct {
    float LowerRange;
    float UpperRange;
    float alpha_13;
    float alpha_11;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
} MlasTanhConstants = {
    -9.0f,
    9.0f,
    -2.76076847742355e-16f,
    2.00018790482477e-13f,
    -8.60467152213735e-11f,
    5.12229709037114e-08f,
    1.48572235717979e-05f,
    6.37261928875436e-04f,
    4.89352455891786e-03f,
    1.19825839466702e-06f,
    1.18534705686654e-04f,
    2.26843463243900e-03f,
    4.89352518554385e-03f,
};

template <>
Status Tanh<float>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  Tensor* Y = context->Output(0, x_shape);

  const float *input = X->template Data<float>();
  float *output = Y->template MutableData<float>();
  size_t numElem = x_shape.Size();

    for (size_t i = 0; i < numElem; i++) {
      float Value = (std::min)(MlasTanhConstants.UpperRange, (std::max)(MlasTanhConstants.LowerRange, input[i]));

      float ValueSquared = Value * Value;

      float p;
      p = ValueSquared * MlasTanhConstants.alpha_13 + MlasTanhConstants.alpha_11;
      p = p * ValueSquared + MlasTanhConstants.alpha_9;
      p = p * ValueSquared + MlasTanhConstants.alpha_7;
      p = p * ValueSquared + MlasTanhConstants.alpha_5;
      p = p * ValueSquared + MlasTanhConstants.alpha_3;
      p = p * ValueSquared + MlasTanhConstants.alpha_1;
      p = p * Value;

      float q;
      q = ValueSquared * MlasTanhConstants.beta_6 + MlasTanhConstants.beta_4;
      q = q * ValueSquared + MlasTanhConstants.beta_2;
      q = q * ValueSquared + MlasTanhConstants.beta_0;

      output[i] = (p / q);
    }

  return Status::OK();
}
}  // namespace onnxruntime
