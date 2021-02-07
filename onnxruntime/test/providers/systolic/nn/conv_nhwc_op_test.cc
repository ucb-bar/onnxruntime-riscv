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

#ifdef SYSTOLIC_FP32

// Has only a single layer of conv
TEST(SystolicConvNHWCTest, ConvNHWCSignedTiny2DTest) {
  OpTester test("Conv_nhwc", 10);

  std::vector<float> X = {110, 35, 111, 107, 5,
                           79, 103, 5, 12, 123,
                           34, 40, 41, 102, 33,
                           117, 109, 73, 51, 123,
                           6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 1, 5, 5};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {1, 1, 1, 1,
                           1, 1, 1, 1,
                           1, 1, 1, 1,
                           1, 1, 1, 1};
  std::vector<int64_t> W_shape = {1, 1, 4, 4};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> expected_vals = {1229, 1173,
                                       1165, 1319};
  std::vector<int64_t> Y_shape = {1, 1, 2, 2};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<float>("x", X_shape, X);
  test.AddInput<float>("w", W_shape, W);
  test.AddInput<float>("B", {1}, {100});

  test.AddOutput<float>("y", Y_shape, expected_vals);

  test.Run();
}

TEST(SystolicConvNHWCTest, ConvNHWCSignedTiny2DTest2) {
  OpTester test("Conv_nhwc", 10);

  std::vector<float> X = {110, 35, 111, 107, 5,
                           79, 103, 5, 12, 123,
                           34, 40, 41, 102, 33,
                           117, 109, 73, 51, 123,
                           6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 1, 5, 5};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {1, 2,
                           3, 4};
  std::vector<int64_t> W_shape = {1, 1, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> expected_vals = {929, 686, 488, 745,
                                      647, 497, 660, 796,
                                      1001, 841, 768, 913,
                                      957, 957, 887, 1174};
  std::vector<int64_t> Y_shape = {1, 1, 4, 4};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<float>("x", X_shape, X);
  test.AddInput<float>("w", W_shape, W);

  test.AddInput<float>("B", {1}, {100});

  test.AddOutput<float>("y", Y_shape, expected_vals);

  test.Run();
}

TEST(SystolicConvNHWCTest, ConvNHWCWithMultipleOutputChannel) {
  OpTester test("Conv_nhwc", 10);

  std::vector<float> X = {110, 35, 111, 107, 5,
                           79, 103, 5, 12, 123,
                           34, 40, 41, 102, 33,
                           117, 109, 73, 51, 123,
                           6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 1, 5, 5};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {1, 2,
                           3, 4,

                           1, 2,
                           3, 4};
  std::vector<int64_t> W_shape = {2, 1, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> expected_vals = {929, 686, 488, 745,
                                      647, 497, 660, 796,
                                      1001, 841, 768, 913,
                                      957, 957, 887, 1174,
                                       
                                       829, 586, 388, 645,
                                       547, 397, 560, 696,
                                       901, 741, 668, 813,
                                       857, 857, 787, 1074};

  std::vector<int64_t> Y_shape = {1, 2, 4, 4};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<float>("x", X_shape, X);
  test.AddInput<float>("w", W_shape, W);

  test.AddInput<float>("B", {2}, {100, 0});
  test.AddOutput<float>("y", Y_shape, expected_vals);

  test.Run();
}

TEST(SystolicConvNHWCTest, ConvNHWCMultipleInputAndOutputChannel) {
  OpTester test("Conv_nhwc", 10);

  std::vector<float> X = {110, 35, 111, 107, 5,
                           79, 103, 5, 12, 123,
                           34, 40, 41, 102, 33,
                           117, 109, 73, 51, 123,
                           6, 126, 56, 111, 111,
                           
                           -61, -41, -113, -92, -72,
                           -94, -104, -84, -53, -95,
                           -30, -81, -102, -44, -48,
                           -62, -53, -108, -25, -72,
                           -58, -57, -39, -22, -29};
  std::vector<int64_t> X_shape = {1, 2, 5, 5};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {1, 2,
                           3, 4,
                           /* */
                           1, 1,
                           1, 1,
                           /*--*/
                           1, 2,
                           3, 4,
                           /* */
                           1, 0,
                           0, 1};

  std::vector<int64_t> W_shape = {2, 2, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> expected_vals = {629, 344, 146, 433,
                                      338, 126, 377, 556,
                                      775, 497, 489, 724,
                                      727, 700, 693, 1026,

                                       664, 461, 222, 458,
                                       372, 191, 432, 595,
                                       818, 552, 541, 697,
                                       738, 765, 657, 1020};

  std::vector<int64_t> Y_shape = {1, 2, 4, 4};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<float>("x", X_shape, X);
  test.AddInput<float>("w", W_shape, W);

  test.AddInput<float>("B", {2}, {100, 0});

  test.AddOutput<float>("y", Y_shape, expected_vals);

  test.Run();
}

TEST(SystolicConvNHWCTest, ConvNHWCBatchTest) {
  OpTester test("Conv_nhwc", 10);

  std::vector<float> X = {110, 35, 111, 107, 5,
                           79, 103, 5, 12, 123,
                           34, 40, 41, 102, 33,
                           117, 109, 73, 51, 123,
                           6, 126, 56, 111, 111,
                           
                           -61, -41, -113, -92, -72,
                           -94, -104, -84, -53, -95,
                           -30, -81, -102, -44, -48,
                           -62, -53, -108, -25, -72,
                           -58, -57, -39, -22, -29};
  std::vector<int64_t> X_shape = {2, 1, 5, 5};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<float> W = {1, 2,
                           3, 4};

  std::vector<int64_t> W_shape = {1, 1, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<float> expected_vals = {929, 686, 488, 745,
                                      647, 497, 660, 796,
                                      1001, 841, 768, 913,
                                      957, 957, 887, 1174,

                                      -741, -815, -661, -675,
                                      -616, -823, -572, -467,
                                      -490, -776, -514, -403,
                                      -470, -496, -263, -251};

  std::vector<int64_t> Y_shape = {2, 1, 4, 4};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<float>("x", X_shape, X);
  test.AddInput<float>("w", W_shape, W);

  test.AddInput<float>("B", {1}, {100});

  test.AddOutput<float>("y", Y_shape, expected_vals);

  test.Run();
}

#endif

}  // namespace test
}  // namespace onnxruntime
