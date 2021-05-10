#include "qlinearconv_relu_fuse.h"

namespace onnxruntime {
namespace systolic {

// Looking for the conv + maxpool fusion? It's in nhwc_transformer.cc

/**
 * Memo: if you ever get a sudden termination after the fusion phase, ensure that your meta_def name is registered
 * For some reason onnxruntime will silently fail if it can't lookup kernel for a fused operator
 * 
 * Also note that for fusion to occur no execution provider must be assignd to the node beforehand
 */

/**
 * NOTE: The way the fused kernels are reigstered is a bit subtle. The fused kernels
 * don't need an explicit schema declaration (because they're fused). Instead,
 * we add an attribute "relu" that we check in the constructor of the kernel.
 * ORT matches based on the op type (which for fused kernels is equal to meta_def->name),
 * so adding extra attrs won't cause any issues with non-fused variants.
 * (And schema validation/shape-inference isn't explicitly done for fused-ops apparently,
 * which makes sense I guess since it's basically just composition of the underlying ops)
 * This way, we can re-use a lot of code and avoid combinatorial explosion as we add more fusion types.
 */

std::unique_ptr<::onnxruntime::IndexedSubGraph::MetaDef> getFusedQlinearConvReluMeta(const Node* qlinearconv, const Node* relu) {
  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = qlinearconv->OpType();
  meta_def->domain = "";
  meta_def->since_version = 10;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->attributes = qlinearconv->GetAttributes();

  ONNX_NAMESPACE::AttributeProto relu_attr;
  relu_attr.set_name("relu");
  relu_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  relu_attr.set_i(1);
  meta_def->attributes["relu"] = relu_attr;

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
  meta_def->type_and_shape_inference_function = qlinearconv->Op()->GetTypeAndShapeInferenceFunction();

  return meta_def;
}

void qlinearconv_relu_fuse::operator()(const onnxruntime::GraphViewer& graph, std::vector<std::unique_ptr<ComputeCapability>>& capabilites) {
  LOGS_DEFAULT(INFO) << "Called into Systolic fuser for QLinearConv + Relu";
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
      capability->sub_graph->SetMetaDef(std::move(meta_def));
    }
  }
  LOGS_DEFAULT(INFO) << "Finished systolic fusing for QLinearConv + Relu";
}

}  // namespace systolic
}  // namespace onnxruntime
