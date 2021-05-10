// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/systolic/systolic_provider_factory.h"
#include <atomic>
#include "systolic_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct SystolicProviderFactory : IExecutionProviderFactory {
  SystolicProviderFactory(bool create_arena, int accelerator_mode) : create_arena_(create_arena), accelerator_mode_(accelerator_mode) {}
  ~SystolicProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
  char accelerator_mode_;
};

std::unique_ptr<IExecutionProvider> SystolicProviderFactory::CreateProvider() {
  SystolicExecutionProviderInfo info;
  info.create_arena = create_arena_;
  info.accelerator_mode = accelerator_mode_;
  return std::make_unique<SystolicExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Systolic(int use_arena, char accelerator_mode) {
  return std::make_shared<onnxruntime::SystolicProviderFactory>(use_arena != 0, accelerator_mode);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Systolic, _In_ OrtSessionOptions* options, int use_arena, char accelerator_mode) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Systolic(use_arena, accelerator_mode));
  return nullptr;
}