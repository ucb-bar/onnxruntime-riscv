// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

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

  SessionOptions session_options;
  session_options.intra_op_param.thread_pool_size = 1;
  test.Run(session_options);
}

}  // namespace test
}  // namespace onnxruntime
