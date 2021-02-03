// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

/**
 * Note that the inputs to the op must be NHWC and weights must be HWIO
 * Make use of the helper OHWI -> HWIO converter since it's easier to 
 * specify inputs in OWHI format when writing test cases
 * 
 * The inputs to this function are weights layed out in OIHW format (with shape in OIHW)
 * and it rewrites the inputs to HWIO
 */
template<typename T>
inline void OIHWtoHWIOconvert(std::vector<T> &w_vals, std::vector<int64_t> &w_shape) {
  const std::vector<T> w_vals_copy = w_vals;
  int OC = w_shape[0];
  int IC = w_shape[1];
  int H =  w_shape[2];
  int W =  w_shape[3];

  for (int oc = 0; oc < OC; oc++) {
    for (int ic = 0; ic < IC; ic++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          w_vals[h * W * IC * OC + w * IC * OC + ic * OC + oc] =
            w_vals_copy[oc * H * W * IC + ic * H * W + h * W + w];
        }
      }
    }
  }

  w_shape = {H, W, IC, OC};
}

/**
 * Similarly, it is easier to manually specify inputs in NCHW format (for pretty printing).
 * Use this helper.
 */
template<typename T>
inline void NCHWtoNHWCconvert(std::vector<T> &vals, std::vector<int64_t> &shape) {
  const std::vector<T> vals_copy = vals;
  int N = shape[0];
  int C = shape[1];
  int H = shape[2];
  int W = shape[3];

  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          vals[((n*H + h)*W + w)*C + c] = vals_copy[((n*C + c)*H + h)*W + w];
        }
      }
    }
  }
  shape = {N, H, W, C};
}

namespace onnxruntime {
namespace test {

namespace {

#if defined(SYSTOLIC_FP32) && defined(ENABLE_TRAINING)

TEST(SystolicConvGradTest, BasicTest) {
  OpTester test("ConvGrad", 9);

  std::vector<float> X = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> X_shape = {1, 1, 3, 3};

  std::vector<float> W = {-1, -2, -3, -4};
  std::vector<int64_t> W_shape = {1, 1, 2, 2};

  std::vector<float> dY = {0.1, 0.2, 0.3, 0.4};
  std::vector<int64_t> dY_shape = {1, 1, 2, 2};

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);

  std::vector<int64_t> dX_shape = {1, 1, 3, 3};
  std::vector<int64_t> dW_shape = {1, 1, 2, 2};
  std::vector<int64_t> dB_shape = {1};

  std::vector<float> expected_dX = {-0.1, -0.4, -0.4, -0.6, -2., -1.6, -0.9, -2.4, -1.6};
  std::vector<float> expected_dW = {3.7, 4.7, 6.7, 7.7};
  std::vector<float> expected_dB = {1};

  test.AddOutput<float>("dX", dX_shape, expected_dX);
  test.AddOutput<float>("dW", dW_shape, expected_dW);
  test.AddOutput<float>("dB", dB_shape, expected_dB);

  test.Run();
}

TEST(SystolicConvGradTest, BasicNHWCTest) {
  OpTester test("ConvGrad_nhwc", 9);

  std::vector<float> X = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> X_shape = {1, 1, 3, 3};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {-1, -2, -3, -4};
  std::vector<int64_t> W_shape = {1, 1, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> dY = {0.1, 0.2, 0.3, 0.4};
  std::vector<int64_t> dY_shape = {1, 1, 2, 2};
  NCHWtoNHWCconvert(dY, dY_shape);

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);

  std::vector<int64_t> dX_shape = {1, 1, 3, 3};
  std::vector<int64_t> dW_shape = {1, 1, 2, 2};
  std::vector<int64_t> dB_shape = {1};

  std::vector<float> expected_dX = {-0.1, -0.4, -0.4, -0.6, -2., -1.6, -0.9, -2.4, -1.6};
  std::vector<float> expected_dW = {3.7, 4.7, 6.7, 7.7};

  OIHWtoHWIOconvert(expected_dW, dW_shape);
  NCHWtoNHWCconvert(expected_dX, dX_shape);

  std::vector<float> expected_dB = {1};

  test.AddOutput<float>("dX", dX_shape, expected_dX);
  test.AddOutput<float>("dW", dW_shape, expected_dW);
  test.AddOutput<float>("dB", dB_shape, expected_dB);

  test.Run();
}

#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
