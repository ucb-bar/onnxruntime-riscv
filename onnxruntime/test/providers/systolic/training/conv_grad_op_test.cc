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
template <typename T>
inline void OIHWtoHWIOconvert(std::vector<T>& w_vals, std::vector<int64_t>& w_shape) {
  const std::vector<T> w_vals_copy = w_vals;
  int OC = w_shape[0];
  int IC = w_shape[1];
  int H = w_shape[2];
  int W = w_shape[3];

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
template <typename T>
inline void NCHWtoNHWCconvert(std::vector<T>& vals, std::vector<int64_t>& shape) {
  const std::vector<T> vals_copy = vals;
  int N = shape[0];
  int C = shape[1];
  int H = shape[2];
  int W = shape[3];

  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          vals[((n * H + h) * W + w) * C + c] = vals_copy[((n * C + c) * H + h) * W + w];
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

TEST(SystolicConvGradTest, BatchSizeTest) {
  OpTester test("ConvGrad", 9);

  std::vector<float> X = {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9,

                          -1, -2, -3,
                          -4, -5, -6,
                          -7, -8, -9};
  std::vector<int64_t> X_shape = {2, 1, 3, 3};

  std::vector<float> W = {-1, -2, -3, -4};
  std::vector<int64_t> W_shape = {1, 1, 2, 2};

  std::vector<float> dY = {0.1, 0.2,
                           0.3, 0.4,

                           1, 0,
                           0, 1};
  std::vector<int64_t> dY_shape = {2, 1, 2, 2};

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);

  std::vector<int64_t> dX_shape = {2, 1, 3, 3};
  std::vector<int64_t> dW_shape = {1, 1, 2, 2};
  std::vector<int64_t> dB_shape = {1};

  std::vector<float> expected_dX = {-0.1, -0.4, -0.4,
                                    -0.6, -2., -1.6,
                                    -0.9, -2.4, -1.6,

                                    -1, -2, 0,
                                    -3, -5, -2,
                                    0, -3, -4};
  std::vector<float> expected_dW = {-2.3, -3.3,
                                    -5.3, -6.3};
  std::vector<float> expected_dB = {3};

  test.AddOutput<float>("dX", dX_shape, expected_dX);
  test.AddOutput<float>("dW", dW_shape, expected_dW);
  test.AddOutput<float>("dB", dB_shape, expected_dB);

  test.Run();
}

TEST(SystolicConvGradTest, MultipleOutputChannel) {
  OpTester test("ConvGrad", 9);

  std::vector<float> X = {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9};
  std::vector<int64_t> X_shape = {1, 1, 3, 3};

  std::vector<float> W = {-1, -2,
                          -3, -4,

                          1, 0,
                          0, 1};
  std::vector<int64_t> W_shape = {2, 1, 2, 2};

  std::vector<float> dY = {0.1, 0.2,
                           0.3, 0.4,

                           3, 1,
                           4, 1};
  std::vector<int64_t> dY_shape = {1, 2, 2, 2};

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);

  std::vector<int64_t> dX_shape = {1, 1, 3, 3};
  std::vector<int64_t> dW_shape = {2, 1, 2, 2};
  std::vector<int64_t> dB_shape = {2};

  std::vector<float> expected_dX = {2.9, 0.6, -0.4,
                                    3.4, 2, -0.6,
                                    -0.9, 1.6, -0.6};
  std::vector<float> expected_dW = {3.7, 4.7,
                                    6.7, 7.7,

                                    26, 35,
                                    53, 62};
  std::vector<float> expected_dB = {1, 9};

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

/**
 * Multiple output channel:
 * lhs matrix
    0.1 0.2 0.3 0.4 
    3 1 4 1 
    rhs matrix
    1 2 4 5 
    2 3 5 6 
    4 5 7 8 
    5 6 8 9 
    out matrix
    3.7 4.7 6.7 7.7 
    26 35 53 62 

    Note that each output channel is handled "independently" and we then accumulate over the batches
 */
TEST(SystolicConvGradTest, WeightMultipleOutputChannelNHWCTest) {
  OpTester test("ConvGrad_nhwc", 9);

  std::vector<float> X = {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9};
  std::vector<int64_t> X_shape = {1, 1, 3, 3};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {-1, -2,
                          -3, -4,

                          1, 0,
                          0, 1};
  std::vector<int64_t> W_shape = {2, 1, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> dY = {0.1, 0.2,
                           0.3, 0.4,

                           3, 1,
                           4, 1};
  std::vector<int64_t> dY_shape = {1, 2, 2, 2};

  NCHWtoNHWCconvert(dY, dY_shape);

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);

  std::vector<int64_t> dX_shape = {1, 1, 3, 3};
  std::vector<int64_t> dW_shape = {2, 1, 2, 2};
  std::vector<int64_t> dB_shape = {2};

  std::vector<float> expected_dX = {2.9, 0.6, -0.4,
                                    3.4, 2, -0.6,
                                    -0.9, 1.6, -0.6};
  std::vector<float> expected_dW = {3.7, 4.7,
                                    6.7, 7.7,

                                    26, 35,
                                    53, 62};

  OIHWtoHWIOconvert(expected_dW, dW_shape);
  NCHWtoNHWCconvert(expected_dX, dX_shape);

  std::vector<float> expected_dB = {1, 9};

  test.AddOutput<float>("dX", dX_shape, expected_dX);
  test.AddOutput<float>("dW", dW_shape, expected_dW);
  test.AddOutput<float>("dB", dB_shape, expected_dB);

  test.Run();
}

TEST(SystolicConvGradTest, BatchSizeNHWCTest) {
  OpTester test("ConvGrad_nhwc", 9);

  std::vector<float> X = {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9,

                          -1, -2, -3,
                          -4, -5, -6,
                          -7, -8, -9};
  std::vector<int64_t> X_shape = {2, 1, 3, 3};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {-1, -2, -3, -4};
  std::vector<int64_t> W_shape = {1, 1, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> dY = {0.1, 0.2,
                           0.3, 0.4,

                           1, 0,
                           0, 1};
  std::vector<int64_t> dY_shape = {2, 1, 2, 2};
  NCHWtoNHWCconvert(dY, dY_shape);

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);

  std::vector<int64_t> dX_shape = {2, 1, 3, 3};
  std::vector<int64_t> dW_shape = {1, 1, 2, 2};
  std::vector<int64_t> dB_shape = {1};

  std::vector<float> expected_dX = {-0.1, -0.4, -0.4,
                                    -0.6, -2., -1.6,
                                    -0.9, -2.4, -1.6,

                                    -1, -2, 0,
                                    -3, -5, -2,
                                    0, -3, -4};
  std::vector<float> expected_dW = {-2.3, -3.3,
                                    -5.3, -6.3};

  OIHWtoHWIOconvert(expected_dW, dW_shape);
  NCHWtoNHWCconvert(expected_dX, dX_shape);

  std::vector<float> expected_dB = {3};

  test.AddOutput<float>("dX", dX_shape, expected_dX);
  test.AddOutput<float>("dW", dW_shape, expected_dW);
  test.AddOutput<float>("dB", dB_shape, expected_dB);

  test.Run();
}

/**
 * lhs matrix
    0.1 0.2 0.3 0.4 
    rhs matrix
    1 2 5 6 
    3 4 7 8 
    9 10 13 14 
    11 12 15 16 
    out matrix
    7.8 8.8 11.8 12.8 
 */
TEST(SystolicConvGradTest, NHWCStrideTest) {
  OpTester test("ConvGrad_nhwc", 9);

  std::vector<float> X = {1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12,
                          13, 14, 15, 16};
  std::vector<int64_t> X_shape = {1, 1, 4, 4};
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

  std::vector<int64_t> dX_shape = {1, 1, 4, 4};
  std::vector<int64_t> dW_shape = {1, 1, 2, 2};
  std::vector<int64_t> dB_shape = {1};

  std::vector<float> expected_dX = {-0.1, -0.2, -0.2, -0.4,
                                    -0.3, -0.4, -0.6, -0.8,
                                    -0.3, -0.6, -0.4, -0.8,
                                    -0.9, -1.2, -1.2, -1.6};
  std::vector<float> expected_dW = {7.8, 8.8, 11.8, 12.8};

  OIHWtoHWIOconvert(expected_dW, dW_shape);
  NCHWtoNHWCconvert(expected_dX, dX_shape);

  std::vector<float> expected_dB = {1};

  test.AddOutput<float>("dX", dX_shape, expected_dX);
  test.AddOutput<float>("dW", dW_shape, expected_dW);
  test.AddOutput<float>("dB", dB_shape, expected_dB);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});

  test.Run();
}

/**
 * lhs matrix
    0.1 0.2 0.3 0.4 
    3 1 4 1 
    rhs matrix
    1 11 2 12 4 14 5 15 
    2 12 3 13 5 15 6 16 
    4 14 5 15 7 17 8 18 
    5 15 6 16 8 18 9 19 
    out matrix
    3.7 13.7 4.7 14.7 6.7 16.7 7.7 17.7 
    26 116 35 125 53 143 62 152 
 */
TEST(SystolicConvGradTest, WeightMultipleInputOutputChannelNHWCTest) {
  OpTester test("ConvGrad_nhwc", 9);

  std::vector<float> X = {1, 2, 3,
                          4, 5, 6,
                          7, 8, 9,

                          11, 12, 13,
                          14, 15, 16,
                          17, 18, 19};
  std::vector<int64_t> X_shape = {1, 2, 3, 3};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {-1, -2,
                          -3, -4,

                          1, 0,
                          0, 1,

                          /***/

                          -11, -12,
                          -13, -14,

                          11, 10,
                          10, 11};
  std::vector<int64_t> W_shape = {2, 2, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> dY = {0.1, 0.2,
                           0.3, 0.4,

                           3, 1,
                           4, 1};
  std::vector<int64_t> dY_shape = {1, 2, 2, 2};

  NCHWtoNHWCconvert(dY, dY_shape);

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("W", W_shape, W);

  std::vector<int64_t> dX_shape = {1, 2, 3, 3};
  std::vector<int64_t> dW_shape = {2, 2, 2, 2};
  std::vector<int64_t> dB_shape = {2};

  std::vector<float> expected_dX = {-33.1, -47.4, -12.4,
                                    -83.6, -116, -27.6,
                                    -52.9, -71.4, -15.6,
                                    33.1, 41.2, 10,
                                    74.3, 94.5, 21.2,
                                    40, 54.3, 11.4};

  std::vector<float> expected_dW = {3.7, 4.7,
                                    6.7, 7.7,
                                    13.7, 14.7,
                                    16.7, 17.7,
                                    26., 35.,
                                    53., 62.,
                                    116., 125.,
                                    143., 152.};

  OIHWtoHWIOconvert(expected_dW, dW_shape);
  NCHWtoNHWCconvert(expected_dX, dX_shape);

  std::vector<float> expected_dB = {1, 9};

  test.AddOutput<float>("dX", dX_shape, expected_dX);
  test.AddOutput<float>("dW", dW_shape, expected_dW);
  test.AddOutput<float>("dB", dB_shape, expected_dB);

  test.Run();
}

// Test the case where we have ((x_size + 2*padding) - w_size) % stride != 0
TEST(SystolicConvGradTest, StridesAndPaddingNHWC) {
  OpTester test("ConvGrad_nhwc", 9);

  std::vector<float> X = {5, 5, 0,
                          3, 5, 4,
                          6, 7, 6};

  std::vector<float> W = {9, 4,
                         7, 7};

  std::vector<float> dY = {6, 4,
                           8, 8};


  test.AddInput<float>("dY", {1, 2, 2, 1}, dY);
  test.AddInput<float>("X", {1, 3, 3, 1}, X);
  test.AddInput<float>("W", {2, 2, 1, 1}, W);


  std::vector<float> expected_dX = {0,  0,  0, 
                                    0, 72, 32, 
                                    0, 56, 56};

  std::vector<float> expected_dW = {40, 32,
                                   56, 48};


  test.AddOutput<float>("dX", {1, 3, 3, 1}, expected_dX);
  test.AddOutput<float>("dW", {2, 2, 1, 1}, expected_dW);

  test.AddAttribute("strides", std::vector<int64_t>{3, 3});
  test.AddAttribute("pads", std::vector<int64_t>{2, 2, 2, 2}); // Beginning and ending for each spatial axis

  test.Run();
}

#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime