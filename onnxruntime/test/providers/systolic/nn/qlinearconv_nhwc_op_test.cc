// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#ifdef SYSTOLIC_INT8

inline void PerformIOHWtoHWIOconversion(std::vector<int8_t>& data, const std::vector<int64_t>& shape) {
    std::vector<int8_t> data_copy = data;
    int OC = shape[0];
    int H = shape[1];
    int W =  shape[2];
    int IC =  shape[3];

    for (int k = 0; k < H * W; k++) {
      for (int ic = 0; ic < IC; ic++) {
        for (int oc = 0; oc < OC; oc++) {
          data[k * IC * OC + ic * OC + oc] = data_copy[oc * H * W * IC + ic * H * W + k];
        }
      }
    }
}

// Has only a single layer of conv
TEST(ConvTest, QLinearConvNHWCSignedTiny2DTest) {
  OpTester test("QLinearConv_nhwc", 10);

  std::vector<int8_t> X = {110, 35, 111, 107, 5, 79, 103, 5, 12, 123, 34, 40, 41, 102, 33, 117, 109, 73, 51, 123, 6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 5, 5, 1};

  std::vector<int8_t> W = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> W_shape = {1, 4, 4, 1};

  std::vector<int8_t> expected_vals = {10, 9, 9, 10};
  std::vector<int64_t> Y_shape = {1, 2, 2, 1};

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

  std::vector<int8_t> X = {110, 35, 111, 107, 5, 79, 103, 5, 12, 123, 34, 40, 41, 102, 33, 117, 109, 73, 51, 123, 6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 5, 5, 1};

  std::vector<int8_t> W = {1, 2, 3, 4};
  std::vector<int64_t> W_shape = {1, 2, 2, 1};

  std::vector<int8_t> expected_vals = {127, 98, 70, 106, 92, 71, 94, 114, 127, 120, 110, 127, 127, 127, 127, 127};
  std::vector<int64_t> Y_shape = {1, 4, 4, 1};

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

TEST(ConvTest, QLinearConvNHWCSignedTiny2DTestWithMultipleOutputChannel) {
  OpTester test("QLinearConv_nhwc", 10);

  std::vector<int8_t> X = {110, 35, 111, 107, 5, 79, 103, 5, 12, 123, 34, 40, 41, 102, 33, 117, 109, 73, 51, 123, 6, 126, 56, 111, 111};
  std::vector<int64_t> X_shape = {1, 5, 5, 1};

  std::vector<int8_t> W = {1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<int64_t> W_shape = {2, 2, 2, 1};
  PerformIOHWtoHWIOconversion(W, W_shape);

  std::vector<int8_t> expected_vals = {127, 118, 98, 84, 70, 55, 106, 92, 92, 78, 71,
                                       57, 94, 80, 114, 99, 127, 127, 120, 106, 110, 95,
                                       127, 116, 127, 122, 127, 122, 127, 112, 127, 127};
  std::vector<int64_t> Y_shape = {1, 4, 4, 2};

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
