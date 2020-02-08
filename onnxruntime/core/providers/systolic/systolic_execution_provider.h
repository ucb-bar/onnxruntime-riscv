// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

constexpr const char* SYSTOLIC = "Systolic";

// Information needed to construct Systolic execution providers.
struct SystolicExecutionProviderInfo {
  bool create_arena{true};
  char accelerator_mode{0};

  explicit SystolicExecutionProviderInfo(bool use_arena, char accelerator_mode)
      : create_arena(use_arena), accelerator_mode(accelerator_mode) {}

  SystolicExecutionProviderInfo() = default;
};

using FuseRuleFn = std::function<void(const onnxruntime::GraphViewer&,
                                      std::vector<std::unique_ptr<ComputeCapability>>&)>;

// Logical device representation.
class SystolicExecutionProvider : public IExecutionProvider {
 public:
  explicit SystolicExecutionProvider(const SystolicExecutionProviderInfo& info)
      : IExecutionProvider{onnxruntime::kSystolicExecutionProvider}, provider_info_(info) {
    DeviceAllocatorRegistrationInfo device_info{OrtMemTypeDefault,
                                                [](int) {
                                                  auto memory_info = onnxruntime::make_unique<OrtMemoryInfo>(SYSTOLIC, OrtAllocatorType::OrtDeviceAllocator);
                                                  return onnxruntime::make_unique<TAllocator>(std::move(memory_info)); },
                                                std::numeric_limits<size_t>::max()};

#ifdef USE_JEMALLOC
#if defined(USE_MIMALLOC)
#error jemalloc and mimalloc should not both be enabled
#endif

    ORT_UNUSED_PARAMETER(info);
    //JEMalloc already has memory pool, so just use device allocator.
    InsertAllocator(
        std::shared_ptr<IArenaAllocator>(
            onnxruntime::make_unique<DummyArena>(device_info.factory(0))));
#else
//Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
#if defined(__amd64__) || defined(_M_AMD64)
    if (info.create_arena) {
      InsertAllocator(CreateAllocator(device_info));
    }
    else
#endif
    {
      ORT_UNUSED_PARAMETER(info);
      InsertAllocator(
          std::shared_ptr<IArenaAllocator>(
              onnxruntime::make_unique<DummyArena>(device_info.factory(0))));
    }
#endif
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;
  char GetAcceleratorMode() const;

 private:
  std::vector<FuseRuleFn> fuse_rules_;
  SystolicExecutionProviderInfo provider_info_;
};
}  // namespace onnxruntime
