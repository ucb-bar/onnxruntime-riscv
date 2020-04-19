// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/systolic/systolic_execution_provider.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(SystolicExecutionProviderTest, MetadataTest) {
  SystolicExecutionProviderInfo info;
  auto provider = onnxruntime::make_unique<SystolicExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, "Systolic");
}
}  // namespace test
}  // namespace onnxruntime
