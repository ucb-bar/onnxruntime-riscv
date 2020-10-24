// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hwacha/hwacha_execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "hwacha_fwd.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {

namespace hwacha {

// Forward declarations of op kernels

static Status RegisterHwachaKernels(KernelRegistry& kernel_registry) {
    static const BuildKernelCreateInfoFn function_table[] = {};

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

struct KernelRegistryAndStatus {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  Status st;
};

KernelRegistryAndStatus GetHwachaKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterHwachaKernels(*ret.kernel_registry);
  return ret;
}

}  // namespace hwacha


void HwachaExecutionProvider::InsertFusedRules(FuseRuleFn rule) {
  fuse_rules_.push_back(rule);
}

std::shared_ptr<KernelRegistry> HwachaExecutionProvider::GetKernelRegistry() const {
  static hwacha::KernelRegistryAndStatus k = onnxruntime::hwacha::GetHwachaKernelRegistry();
  //throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

std::unique_ptr<IDataTransfer> HwachaExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<CPUDataTransfer>();
}

std::vector<std::unique_ptr<ComputeCapability>>
HwachaExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
      result = IExecutionProvider::GetCapability(graph, kernel_registries);

  for (auto& rule : fuse_rules_) {
    rule(graph, result);
  }
  return result;
}

void HwachaExecutionProvider::SetupFusedRules() {}

}  // namespace onnxruntime
