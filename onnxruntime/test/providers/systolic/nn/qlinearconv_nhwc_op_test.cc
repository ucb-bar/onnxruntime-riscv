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
inline void OIHWtoHWIOconvert(std::vector<int8_t> &w_vals, std::vector<int64_t> &w_shape) {
  const std::vector<int8_t> w_vals_copy = w_vals;
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
inline void NCHWtoNHWCconvert(std::vector<int8_t> &vals, std::vector<int64_t> &shape) {
  const std::vector<int8_t> vals_copy = vals;
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

#ifdef SYSTOLIC_INT8

// Has only a single layer of conv
TEST(ConvTest, QLinearConvNHWCSignedTiny2DTest) {
  OpTester test("QLinearConv_nhwc", 10);

  std::vector<int8_t> X = {110, 35, 111, 107, 5,
                           79, 103, 5, 12, 123,
                           34, 40, 41, 102, 33,
                           117, 109, 73, 51, 123,
                           6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 1, 5, 5};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<int8_t> W = {1, 1, 1, 1,
                           1, 1, 1, 1,
                           1, 1, 1, 1,
                           1, 1, 1, 1};
  std::vector<int64_t> W_shape = {1, 1, 4, 4};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<int8_t> expected_vals = {10, 9,
                                       9, 10};
  std::vector<int64_t> Y_shape = {1, 1, 2, 2};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<int8_t>("x", X_shape, X);
  test.AddInput<float>("x_scale", {}, {1});
  test.AddInput<int8_t>("x_zero_point", {}, {0});

  test.AddInput<int8_t>("w", W_shape, W);
  test.AddInput<float>("w_scale", {}, {1});
  test.AddInput<int8_t>("w_zero_point", {}, {0});

  test.AddInput<float>("y_scale", {}, {128});
  test.AddInput<int8_t>("y_zero_point", {}, {0});

  test.AddInput<int32_t>("B", {1}, {100});

  test.AddOutput<int8_t>("y", Y_shape, expected_vals);

  test.Run();
}

TEST(ConvTest, QLinearConvNHWCSignedTiny2DTest2) {
  OpTester test("QLinearConv_nhwc", 10);

  std::vector<int8_t> X = {110, 35, 111, 107, 5,
                           79, 103, 5, 12, 123,
                           34, 40, 41, 102, 33,
                           117, 109, 73, 51, 123,
                           6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 1, 5, 5};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<int8_t> W = {1, 2,
                           3, 4};
  std::vector<int64_t> W_shape = {1, 1, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<int8_t> expected_vals = {127, 98, 70, 106,
                                       92, 71, 94, 114,
                                       127, 120, 110, 127,
                                       127, 127, 127, 127};
  std::vector<int64_t> Y_shape = {1, 1, 4, 4};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<int8_t>("x", X_shape, X);
  test.AddInput<float>("x_scale", {}, {0.01});
  test.AddInput<int8_t>("x_zero_point", {}, {0});

  test.AddInput<int8_t>("w", W_shape, W);
  test.AddInput<float>("w_scale", {}, {1});
  test.AddInput<int8_t>("w_zero_point", {}, {0});

  test.AddInput<float>("y_scale", {}, {0.07});
  test.AddInput<int8_t>("y_zero_point", {}, {0});

  test.AddInput<int32_t>("B", {1}, {100});

  test.AddOutput<int8_t>("y", Y_shape, expected_vals);

  test.Run();
}

TEST(ConvTest, QLinearConvNHWCWithMultipleOutputChannel) {
  OpTester test("QLinearConv_nhwc", 10);

  std::vector<int8_t> X = {110, 35, 111, 107, 5,
                           79, 103, 5, 12, 123,
                           34, 40, 41, 102, 33,
                           117, 109, 73, 51, 123,
                           6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 1, 5, 5};
  NCHWtoNHWCconvert(X, X_shape);

  std::vector<int8_t> W = {1, 2,
                           3, 4,

                           1, 2,
                           3, 4};
  std::vector<int64_t> W_shape = {2, 1, 2, 2};
  OIHWtoHWIOconvert(W, W_shape);

  std::vector<int8_t> expected_vals = {127, 98, 70, 106,
                                       92, 71, 94, 114,
                                       127, 120, 110, 127,
                                       127, 127, 127, 127,
                                       
                                       118, 84, 55, 92,
                                       78, 57, 80, 99,
                                       127, 106, 95, 116,
                                       122, 122, 112, 127};

  std::vector<int64_t> Y_shape = {1, 2, 4, 4};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<int8_t>("x", X_shape, X);
  test.AddInput<float>("x_scale", {}, {0.01});
  test.AddInput<int8_t>("x_zero_point", {}, {0});

  test.AddInput<int8_t>("w", W_shape, W);
  test.AddInput<float>("w_scale", {}, {1});
  test.AddInput<int8_t>("w_zero_point", {}, {0});

  test.AddInput<float>("y_scale", {}, {0.07});
  test.AddInput<int8_t>("y_zero_point", {}, {0});

  test.AddInput<int32_t>("B", {2}, {100, 0});

  test.AddOutput<int8_t>("y", Y_shape, expected_vals);

  test.Run();
}

TEST(ConvTest, QLinearConvNHWCMultipleInputAndOutputChannel) {
  OpTester test("QLinearConv_nhwc", 10);

  std::vector<int8_t> X = {110, 35, 111, 107, 5,
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

  std::vector<int8_t> W = {1, 2,
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

  std::vector<int8_t> expected_vals = {90, 49, 21, 62,
                                       48, 18, 54, 79,
                                       111, 71, 70, 103,
                                       104, 100, 99, 127,

                                       95, 66, 32, 65,
                                       53, 27, 62, 85,
                                       117, 79, 77, 100,
                                       105, 109, 94, 127};

  std::vector<int64_t> Y_shape = {1, 2, 4, 4};
  NCHWtoNHWCconvert(expected_vals, Y_shape);

  test.AddInput<int8_t>("x", X_shape, X);
  test.AddInput<float>("x_scale", {}, {0.01});
  test.AddInput<int8_t>("x_zero_point", {}, {0});

  test.AddInput<int8_t>("w", W_shape, W);
  test.AddInput<float>("w_scale", {}, {1});
  test.AddInput<int8_t>("w_zero_point", {}, {0});

  test.AddInput<float>("y_scale", {}, {0.07});
  test.AddInput<int8_t>("y_zero_point", {}, {0});

  test.AddInput<int32_t>("B", {2}, {100, 0});

  test.AddOutput<int8_t>("y", Y_shape, expected_vals);

  test.Run();
}

#endif

}  // namespace test
}  // namespace onnxruntime
