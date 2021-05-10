#include "qlinearadd_relu_fuse.h"

namespace onnxruntime {
namespace systolic {

/**
 * Memo: if you ever get a sudden termination after the fusion phase, ensure that your meta_def name is registered
 * For some reason onnxruntime will silently fail if it can't lookup kernel for a fused operator
 * 
 * Also note that for fusion to occur no execution provider must be assignd to the node beforehand
 */
std::unique_ptr<::onnxruntime::IndexedSubGraph::MetaDef> getFusedQlinearAddReluMeta(const Node* qlinearadd, const Node* relu) {
  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = qlinearadd->OpType();
  meta_def->domain = kMSDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->attributes = qlinearadd->GetAttributes(); // Add normally has no attributes, but may as well

  ONNX_NAMESPACE::AttributeProto relu_attr;
  relu_attr.set_name("relu");
  relu_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  relu_attr.set_i(1);
  meta_def->attributes["relu"] = relu_attr;

  qlinearadd->ForEachWithIndex(qlinearadd->InputDefs(), [&meta_def](const NodeArg& arg, size_t index) {
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
  meta_def->type_and_shape_inference_function = qlinearadd->Op()->GetTypeAndShapeInferenceFunction();

  return meta_def;
}

void qlinearadd_relu_fuse::operator()(const onnxruntime::GraphViewer& graph, std::vector<std::unique_ptr<ComputeCapability>>& capabilites) {
  LOGS_DEFAULT(INFO) << "Called into Systolic fuser for QLinearAdd + Relu";
  for (const auto& capability : capabilites) {
    // Check that we haven't already fused this node
    if (capability->sub_graph->nodes.size() != 1) {
      continue;
    }
    const Node* node = graph.GetNode(capability->sub_graph->nodes[0]);
    if (node->OpType() != "QLinearAdd") {
      continue;
    }

    // We can fuse if the only immediate downstream is a ReLU
    if (node->GetOutputEdgesCount() == 1 && node->OutputNodesBegin()->OpType() == "QLinearRelu") {
      auto next_node = node->OutputNodesBegin();
      LOGS_DEFAULT(INFO) << "Fusing " << node->OpType() << " and " << next_node->OpType();
      capability->sub_graph->nodes.push_back(next_node->Index());
      auto meta_def = getFusedQlinearAddReluMeta(node, next_node.operator->());
      capability->sub_graph->SetMetaDef(std::move(meta_def));
    }
  }
  LOGS_DEFAULT(INFO) << "Finished systolic fusing for QLinearAdd + Relu";
}

}  // namespace systolic
}  // namespace onnxruntime
