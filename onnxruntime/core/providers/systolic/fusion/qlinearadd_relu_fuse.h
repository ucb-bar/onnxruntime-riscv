
#pragma once

#include "core/common/common.h"
#include "core/graph/model.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
namespace systolic {

class qlinearadd_relu_fuse {
public:
    void operator()(const onnxruntime::GraphViewer& graph, std::vector<std::unique_ptr<ComputeCapability>>& capabilites);
};

}  // namespace systolic
}  // namespace onnxruntime
