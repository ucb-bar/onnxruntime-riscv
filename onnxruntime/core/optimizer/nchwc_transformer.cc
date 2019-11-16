// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nchwc_transformer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status NchwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(modified);
  ORT_UNUSED_PARAMETER(graph_level);
  ORT_UNUSED_PARAMETER(logger);
  return Status::OK();
}

}  // namespace onnxruntime
