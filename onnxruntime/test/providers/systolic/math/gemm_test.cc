// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

#ifdef SYSTOLIC_FP32

void SystolicTestGemmNoTrans(bool b_is_initializer) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f), b_is_initializer);
  test.AddInput<float>("C", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(SystolicGemmOpTest, GemmNoTrans) {
  SystolicTestGemmNoTrans(false);
}

// NNAPI EP requires weight to be an initializer
TEST(SystolicGemmOpTest, GemmNoTransBIsInitializer) {
  SystolicTestGemmNoTrans(true);
}


static void TestGemmBroadcast(bool b_is_initializer) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f), b_is_initializer);
  test.AddInput<float>("C", {3}, std::vector<float>{1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 12.0f, 13.0f,
                         -9.0f, -8.0f, -7.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO : Temporarily disabled due to accuracy issues
#else
  test.Run();
#endif
}

TEST(SystolicGemmOpTest, GemmBroadcast) {
  TestGemmBroadcast(false);
}

TEST(SystolicGemmOpTest, GemmBroadcastBIsInitializer) {
  TestGemmBroadcast(true);
}

static void TestGemmTrans(bool b_is_initializer) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)1);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {4, 2},
                       {1.0f, -1.0f,
                        2.0f, -2.0f,
                        3.0f, -3.0f,
                        4.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f), b_is_initializer);
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32) || defined(OPENVINO_CONFIG_MYRIAD)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.Run();
#endif
}

TEST(SystolicGemmOpTest, GemmTrans) {
  TestGemmTrans(false);
}

TEST(SystolicGemmOpTest, GemmTransBIsInitializer) {
  TestGemmTrans(true);
}

// NNAPI EP's GEMM only works as A*B', add case only B is transposed
TEST(SystolicGemmOpTest, GemmTransB) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {3, 4}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.Run();
#endif
}

TEST(SystolicGemmOpTest, GemmAlphaBeta) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 2.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {7.0f, 7.0f, 7.0f,
                         -3.0f, -3.0f, -3.0f});
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Seg fault in parser
#endif
}

TEST(SystolicGemmOpTest, GemmNaN) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Seg fault in parser
}

TEST(SystolicGemmOpTest, GemmScalarBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {1}, std::vector<float>{1.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(SystolicGemmOpTest, Gemm2DBroadcast_1) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 1}, std::vector<float>{1.0f, 2.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -8.0f, -8.0f, -8.0f});
  test.Run();
}

TEST(SystolicGemmOpTest, Gemm2DBroadcast_2) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  // Same as GemmBroadcast, but adding the unnecessary second dimension.
  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {1, 3}, std::vector<float>{1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 12.0f, 13.0f,
                         -9.0f, -8.0f, -7.0f});
  test.Run();
}

TEST(SystolicGemmOpTest, GemmFalseBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 3}, std::vector<float>{1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -8.0f, -8.0f, -8.0f});
  test.Run();
}

TEST(SystolicGemmOpTest, GemmEmptyTensor) {
  OpTester test("Gemm");

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {0, 4},
                       {});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {0, 3},
                        {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kDnnlExecutionProvider});  //TensorRT: doesn't support dynamic shape yet
}

TEST(SystolicGemmOpTest, GemmNoBiasOpset11) {
  OpTester test("Gemm", 11);

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f});
  // tensorRT don't seem to support missing bias
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SystolicGemmOpTest, GemmWithAlphaOpset11) {
  OpTester test("Gemm", 11);

  test.AddAttribute("alpha", 2.0f);

  test.AddInput<float>("A", {2, 2},
                       {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<float>("B", {2, 2}, std::vector<float>(4, 1.0f));
  test.AddOutput<float>("Y", {2, 2},
                        {6.0f, 6.0f, 14.0f, 14.0f});
  // tensorRT don't seem to support missing bias
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(SystolicGemmOpTest, GemmTransA) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)1);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 2.0f);

  test.AddInput<float>("A", {4, 2},
                       {1, -1,
                       2, -2,
                       3, -3,
                       4, -4});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {7.0f, 7.0f, 7.0f,
                         -3.0f, -3.0f, -3.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Seg fault in parser
}

TEST(SystolicGemmOpTest, GemmTransBHarder) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 2.0f);

  test.AddInput<float>("A", {4, 2},
                       {1, -1,
                       2, -2,
                       3, -3,
                       4, -4});
  test.AddInput<float>("B", {3, 2}, {1, 4,
                                     2, 5,
                                     3, 6});
  test.AddInput<float>("C", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddOutput<float>("Y", {4, 3},
                        {0.5, 0.5, 0.5,
                        -1, -1, -1,
                        -2.5, -2.5, -2.5,
                        -4, -4, -4});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Seg fault in parser
}

#endif

}  // namespace test
}  // namespace onnxruntime
