// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace systolic {

Status RegisterSystolicContribKernels(KernelRegistry& kernel_registry);

}  // namespace systolic
}  // namespace onnxruntime