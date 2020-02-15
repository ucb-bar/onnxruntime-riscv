#include "qlinearconv_relu_fuse.h"

namespace onnxruntime {
namespace systolic {

std::unique_ptr<::onnxruntime::IndexedSubGraph::MetaDef> getFusedQlinearConvReluMeta(const Node* qlinearconv, const Node* relu) {
  auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "Fused_QLinearConv_Relu";
  meta_def->domain = "";
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->attributes = qlinearconv->GetAttributes();

  qlinearconv->ForEachWithIndex(qlinearconv->InputDefs(), [&meta_def](const NodeArg& arg, size_t index) {
    ORT_UNUSED_PARAMETER(index);
    LOGS_DEFAULT(INFO) << "\tInput name: " << arg.Name();
    meta_def->inputs.push_back(arg.Name());
    return common::Status::OK();
  });

  relu->ForEachWithIndex(relu->OutputDefs(), [&meta_def](const NodeArg& arg, size_t index) {
    ORT_UNUSED_PARAMETER(index);
    LOGS_DEFAULT(INFO) << "\tOutput name: " << arg.Name();
    meta_def->outputs.push_back(arg.Name());
    return common::Status::OK();
  });

  return meta_def;
}

void qlinearconv_relu_fuse::operator()(const onnxruntime::GraphViewer& graph, std::vector<std::unique_ptr<ComputeCapability>>& capabilites) {
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(capabilites);
  LOGS_DEFAULT(INFO) << "Called into Systolic fuser";
  for (const auto& capability : capabilites) {
    // Check that we haven't already fused this node
    if (capability->sub_graph->nodes.size() != 1) {
      continue;
    }
    const Node* node = graph.GetNode(capability->sub_graph->nodes[0]);
    if (node->OpType() != "QLinearConv") {
      continue;
    }

    // We can fuse if the only immediate downstream is a ReLU
    if (node->GetOutputEdgesCount() == 1 && node->OutputNodesBegin()->OpType() == "QLinearRelu") {
      auto next_node = node->OutputNodesBegin();
      LOGS_DEFAULT(INFO) << "Fusing " << node->OpType() << " and " << next_node->OpType();
      capability->sub_graph->nodes.push_back(next_node->Index());
      auto meta_def = getFusedQlinearConvReluMeta(node, next_node.operator->());
      capability->sub_graph->SetMetaDef(meta_def);
    }
  }
  LOGS_DEFAULT(INFO) << "Finished systolic fusing";
}

}  // namespace systolic
}  // namespace onnxruntime
