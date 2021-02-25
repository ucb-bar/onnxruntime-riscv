// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class SystolicNhwcTransformer

Transformer that optimizes the graph by using NHWC nodes instead of NCHW nodes
and inserts nodes to reorder tensors as needed. This is meant for Systolic.
*/
class SystolicNhwcTransformer : public GraphTransformer {
 public:
  SystolicNhwcTransformer(bool pretraining_pass = false) noexcept : GraphTransformer("SystolicNhwcTransformer") {
    this->pretraining_pass_ = pretraining_pass;
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  bool pretraining_pass_;
};

}  // namespace onnxruntime
