#include "qlinearconv_relu_fuse.h"

namespace onnxruntime {
namespace systolic {

void qlinearconv_relu_fuse::operator()(const onnxruntime::GraphViewer& graph, std::vector<std::unique_ptr<ComputeCapability>>& capabilites) {
 ORT_UNUSED_PARAMETER(graph);
 ORT_UNUSED_PARAMETER(capabilites);   
 printf("Called into fuser\n");
 for (const auto& capability : capabilites) {
     const Node* node = graph.GetNode(capability->sub_graph->nodes[0]);
     printf("%s -> ", node->OpType().c_str());
     for (auto next_node = node->OutputNodesBegin(); next_node != node->OutputNodesEnd(); ++next_node) {
         printf("%s, ", next_node->OpType().c_str());
        if (next_node->OpType() == "QLinearRelu") {
            printf("Fusing %s and %s",  node->OpType().c_str(), next_node->OpType().c_str());
            capability->sub_graph->nodes.push_back(next_node->Index());

            auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
            meta_def->name = "Fused_QLinearConv_Relu";
            meta_def->domain = "";
            meta_def->since_version = 1;
            meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

            node->ForEachWithIndex(node->InputDefs(), [&meta_def](const NodeArg& arg, size_t index) {
                ORT_UNUSED_PARAMETER(index);
                printf("Input name: %s\n", arg.Name().c_str());
                meta_def->inputs.push_back(arg.Name());
                return common::Status::OK();
            });

            next_node->ForEachWithIndex(next_node->OutputDefs(), [&meta_def](const NodeArg& arg, size_t index) {
                ORT_UNUSED_PARAMETER(index);
                printf("Output name: %s\n", arg.Name().c_str());
                meta_def->outputs.push_back(arg.Name());
                return common::Status::OK();
            });

            capability->sub_graph->SetMetaDef(meta_def);

            break;
        }

     }
     printf("\n");
 }

 printf("Finished fusing\n");
}

}  // namespace systolic
}  // namespace onnxruntime
