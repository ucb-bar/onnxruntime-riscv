// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "xtensor/xadapt.hpp"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

class NhwcTransformerImpl {
 public:
  NhwcTransformerImpl(Graph& graph) noexcept : graph_(graph) {}

  void Transform(Node& node);
  void Finalize(bool& modified);

 private:
  // Associate the following state with each created NHWC output keyed off the
  // original NodeArg.
  struct NhwcArgument {
    // Stores the node that generated the NHWC output.
    Node& output_node_;

    // Stores the NodeArg that represents the NHWC output.
    NodeArg* nhwc_arg_;

    // Stores the original number of uses for the original NodeArg. Edges are
    // removed from the graph as nodes are converted to NHWC form.
    const size_t starting_original_uses_;

    // Stores the remaining number of uses for the original NodeArg. The count
    // is decremented as uses are converted to NHWC format. Nodes are inserted
    // to reorder the output if this count is non-zero.
    size_t remaining_original_uses_;

    NhwcArgument(Node& output_node, NodeArg* output_nhwc_arg, size_t original_uses)
        : output_node_(output_node),
          nhwc_arg_(output_nhwc_arg),
          starting_original_uses_(original_uses),
          remaining_original_uses_(original_uses) {
    }
  };

  size_t RemoveOutputEdges(Node& node);
  void CreateNhwcArgument(Node& node, Node& nhwc_node, const std::string& basename);
  void InsertReorderInput(Node& node);

  void TransformQLinearConv(Node& node);

  Graph& graph_;

  // Stores a queue of nodes to be removed after walking through the graph.
  std::deque<NodeIndex> removed_nodes_;

  // Stores a mapping from the original NodeArg outputs to the NHWC variants
  // created inside this graph transform.
  std::unordered_map<NodeArg*, std::unique_ptr<NhwcArgument>> nhwc_args_;

  // Stores a mapping of NodeArg inputs that have already been reordered, so
  // multiple nodes can share the NHWC input.
  std::unordered_map<NodeArg*, NodeArg*> reorder_inputs_;

  // Stores a mapping of NodeArg filters that have already been reordered, so
  // multiple nodes can share the NHWC filter.
  std::unordered_map<NodeArg*, NodeArg*> filters_transposed;
};

size_t NhwcTransformerImpl::RemoveOutputEdges(Node& node) {
  size_t output_edges_count = node.GetOutputEdgesCount();
  if (output_edges_count > 0) {
    graph_utils::RemoveNodeOutputEdges(graph_, node);
  }
  // Bias the edge count to handle the case of a node that produces a graph
  // output.
  if (!graph_.GetNodeOutputsInGraphOutputs(node).empty()) {
    output_edges_count++;
  }
  return output_edges_count;
}

void NhwcTransformerImpl::CreateNhwcArgument(Node& node,
                                               Node& nhwc_node, const std::string& basename) {
  size_t original_uses = RemoveOutputEdges(node);

  // Create a new NodeArg to track the output from the NHWC node.
  auto& output_defs = nhwc_node.MutableOutputDefs();
  auto* output_original_arg = output_defs[0];
  std::string output_reorder_def_name = graph_.GenerateNodeArgName(basename + "_nhwc");
  auto* output_nhwc_arg = &graph_.GetOrCreateNodeArg(output_reorder_def_name, nullptr);
  nhwc_args_[output_original_arg] =
      onnxruntime::make_unique<NhwcArgument>(nhwc_node, output_nhwc_arg, original_uses);
  output_defs[0] = output_nhwc_arg;
}

void NhwcTransformerImpl::InsertReorderInput(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto* input_original_arg = input_defs[0];

  auto it = reorder_inputs_.find(input_original_arg);
  if (it == reorder_inputs_.end()) {
    std::string input_reorder_def_name = graph_.GenerateNodeArgName("NHWCreorder");
    auto* input_nhwc_arg = &graph_.GetOrCreateNodeArg(input_reorder_def_name, nullptr);
    reorder_inputs_[input_original_arg] = input_nhwc_arg;
    Node& reorder_input_node = graph_.AddNode(graph_.GenerateNodeName("ReorderToNHWC"),
                                              "Transpose",
                                              "ReorderToNHWC",
                                              {input_original_arg},
                                              {input_nhwc_arg},
                                              nullptr,
                                              kOnnxDomain);
    reorder_input_node.AddAttribute("perm",  std::vector<int64_t>({0, 2, 3, 1}));
    reorder_input_node.SetExecutionProviderType(kCpuExecutionProvider);
    input_defs[0] = input_nhwc_arg;
  } else {
    input_defs[0] = it->second;
  }
}

void NhwcTransformerImpl::TransformQLinearConv(Node& node) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Require that the weights tensor be static, and has exactly 4 dims
  const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph_, *input_defs[3]) ||
      !graph_.GetInitializedTensor(input_defs[3]->Name(), conv_W_tensor_proto) ||
      (conv_W_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) ||
      (conv_W_tensor_proto->dims_size() != 4)) {
    printf("133\n");
    return;
  }

  const int64_t output_channels = conv_W_tensor_proto->dims(0);
  const int64_t input_channels = conv_W_tensor_proto->dims(1);
  const int64_t kernel_size = conv_W_tensor_proto->dims(2) * conv_W_tensor_proto->dims(3);
  const int64_t kernel_dim = input_channels * kernel_size;

  int64_t group_count;
  const auto* group_attr = graph_utils::GetNodeAttribute(node, "group");
  if (group_attr != nullptr && utils::HasInt(*group_attr)) {
    group_count = group_attr->i();
  } else {
    group_count = 1;
  }

  NodeArg* nhwc_conv_W_arg;
  auto filters_it = filters_transposed.find(input_defs[3]);
  if (filters_it != filters_transposed.end()) {
    // Reuse the existing NodeArg.
    nhwc_conv_W_arg = filters_it->second;
  } else {
    Initializer conv_W{*conv_W_tensor_proto, graph_.ModelPath()};
    
    std::vector<int8_t> reordered_filter(conv_W.size());
    auto a = xt::adapt(conv_W.data<int8_t>(), conv_W.size(), xt::no_ownership(), conv_W.dims());
    auto tr = xt::flatten(xt::transpose(a, {0, 2, 3, 1}));

    for (int group_id = 0; group_id < group_count; group_id++) {
      int idx_base = group_id * (output_channels / group_count) * kernel_dim;
      
      for (int i = 0; i < kernel_dim; i++) {
        for (int j = 0; j < output_channels / group_count; j++) {
          reordered_filter[idx_base + (i*(output_channels/group_count) + j)] = tr(idx_base + (j*kernel_dim + i));
        }
      }
    }
   
    
    ONNX_NAMESPACE::TensorProto nhwc_conv_W_tensor_proto;
    nhwc_conv_W_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
    nhwc_conv_W_tensor_proto.set_name(graph_.GenerateNodeArgName(input_defs[3]->Name() + "_nhwc"));
    nhwc_conv_W_tensor_proto.set_raw_data(reordered_filter.data(), reordered_filter.size() * sizeof(int8_t));

    nhwc_conv_W_tensor_proto.add_dims(conv_W.dims()[0]);
    nhwc_conv_W_tensor_proto.add_dims(conv_W.dims()[2]);
    nhwc_conv_W_tensor_proto.add_dims(conv_W.dims()[3]);
    nhwc_conv_W_tensor_proto.add_dims(conv_W.dims()[1]);

    nhwc_conv_W_arg = &graph_utils::AddInitializer(graph_, nhwc_conv_W_tensor_proto);
    filters_transposed.emplace(input_defs[3], nhwc_conv_W_arg);
  }

  // Create the replacement node.
  std::string nhwc_node_name = graph_.GenerateNodeName(output_defs[0]->Name() + "_nhwc");
  Node& nhwc_node = graph_.AddNode(nhwc_node_name,
                                    node.OpType() + "_nhwc",
                                    nhwc_node_name,
                                    input_defs,
                                    output_defs,
                                    &node.GetAttributes(),
                                    kOnnxDomain);
  nhwc_node.SetExecutionProviderType(kSystolicExecutionProvider);

  nhwc_node.MutableInputDefs()[3] = nhwc_conv_W_arg;

  if (input_defs.size() == 9) {
    nhwc_node.MutableInputDefs()[8] = input_defs[8];
  }

  // Reorder the input if needed
  auto it = nhwc_args_.find(input_defs[0]);
  if (it == nhwc_args_.end()) {
    InsertReorderInput(nhwc_node);
  } else {
    auto* nhwc_input = it->second.get();
    nhwc_node.MutableInputDefs()[0] = nhwc_input->nhwc_arg_;
    nhwc_input->remaining_original_uses_--;
  }

  CreateNhwcArgument(node, nhwc_node, output_defs[0]->Name());
  removed_nodes_.push_front(node.Index());
}



void NhwcTransformerImpl::Transform(Node& node) {
  if (node.OpType() == "QLinearConv" || node.OpType() == "Fused_QLinearConv_Relu") {
    printf("TRANSFORMING NODE %s\n", node.OpType().c_str());
    TransformQLinearConv(node);
  }

  // The node may not match any of the checks above or may not have been
  // transformed for other reasons such as unsupported attributes or alignment.
  // However, the node may still use an input that has been produced by a NHWC
  // node. Finalize() walks through the list of NHWC outputs and inserts the
  // needed reorder operations to ensure that these inputs remain in NCHW
  // format.
}

void NhwcTransformerImpl::Finalize(bool& modified) {
  // Create ReorderOutput nodes for any NHWC outputs that still have uses with
  // the original tensor format.
  for (auto& nhwc_output : nhwc_args_) {
    if (nhwc_output.second->remaining_original_uses_ > 0) {
      auto* output_original_arg = nhwc_output.first;
      auto* output_nhwc_arg = nhwc_output.second->nhwc_arg_;
      Node& reorder_output_node = graph_.AddNode(graph_.GenerateNodeName("ReorderToNCHW"),
                                                 "Transpose",
                                                 "ReorderToNCHW",
                                                 {output_nhwc_arg},
                                                 {output_original_arg},
                                                 nullptr,
                                                 kOnnxDomain);
      reorder_output_node.SetExecutionProviderType(kCpuExecutionProvider);
      reorder_output_node.AddAttribute("perm", std::vector<int64_t>({0, 3, 1, 2}));
    }
  }

  for (auto index : removed_nodes_) {
    graph_.RemoveNode(index);
  }

  if (!removed_nodes_.empty()) {
    modified = true;
  }
}

Status NhwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  NhwcTransformerImpl impl(graph);
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    if (node.GetExecutionProviderType() == kSystolicExecutionProvider) {
      impl.Transform(node);
    }
  }
  impl.Finalize(modified);
  return Status::OK();
}

}  // namespace onnxruntime
