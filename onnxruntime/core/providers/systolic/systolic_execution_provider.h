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
    SetupFusedRules();

    bool create_arena = info.create_arena;

#ifdef USE_JEMALLOC
#if defined(USE_MIMALLOC_ARENA_ALLOCATOR) || defined(USE_MIMALLOC_STL_ALLOCATOR)
#error jemalloc and mimalloc should not both be enabled
#endif
    //JEMalloc already has memory pool, so just use device allocator.
    create_arena = false;
#elif !(defined(__amd64__) || defined(_M_AMD64))
    //Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
    create_arena = false;
#endif

  AllocatorCreationInfo device_info{
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(SYSTOLIC, OrtAllocatorType::OrtDeviceAllocator));
      },
      0,
      create_arena};

    InsertAllocator(CreateAllocator(device_info));
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;
  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  char GetAcceleratorMode() const;
  void InsertFusedRules(FuseRuleFn rule);
  void SetupFusedRules();

 private:
  std::vector<FuseRuleFn> fuse_rules_;
  SystolicExecutionProviderInfo provider_info_;
};
}  // namespace onnxruntime
