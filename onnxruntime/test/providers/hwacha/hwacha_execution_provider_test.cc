// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hwacha/hwacha_execution_provider.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(HwachaExecutionProviderTest, MetadataTest) {
  HwachaExecutionProviderInfo info;
  auto provider = onnxruntime::make_unique<HwachaExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, "Hwacha");
}
}  // namespace test
}  // namespace onnxruntime
