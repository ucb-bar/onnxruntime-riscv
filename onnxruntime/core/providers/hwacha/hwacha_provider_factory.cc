// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hwacha/hwacha_provider_factory.h"
#include <atomic>
#include "hwacha_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct HwachaProviderFactory : IExecutionProviderFactory {
  HwachaProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~HwachaProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> HwachaProviderFactory::CreateProvider() {
  HwachaExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return onnxruntime::make_unique<HwachaExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Hwacha(int use_arena) {
  return std::make_shared<onnxruntime::HwachaProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Hwacha, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Hwacha(use_arena));
  return nullptr;
}