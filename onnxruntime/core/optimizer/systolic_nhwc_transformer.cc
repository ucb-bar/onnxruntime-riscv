// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/systolic_nhwc_transformer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

class SystolicNhwcTransformerImpl {
 public:
  SystolicNhwcTransformerImpl(Graph& graph, bool pretraining_pass) noexcept : graph_(graph), pretraining_pass_(pretraining_pass) {}

  void Transform(Node& node, const logging::Logger& logger);
  void Finalize(bool& modified, const logging::Logger& logger);

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

  size_t RemoveOutputEdges(Node& node, int idx);
  void CreateNhwcArgument(Node& node, Node& nhwc_node, const std::string& basename);
  void FuseNhwcArgument(Node& node, const NhwcArgument& nhwc_arg);
  void InsertReorderInput(Node& node);

  template <typename T, ONNX_NAMESPACE::TensorProto_DataType ONNX_T>
  void TransformConv(Node& node, const logging::Logger& logger, uint32_t weightIdx, uint32_t biasIdx);
  void TransformQLinearAdd(Node& node, const logging::Logger& logger);
  void TransformMaxPool(Node& node, const logging::Logger& logger);
  bool FuseMaxPoolWithConv(Node& node, const logging::Logger& logger);
  void TransformPassThrough(Node& node, const logging::Logger& logger);

  bool ShouldTransform(Node& node, const logging::Logger& logger);

  Graph& graph_;

  // Whether we should convert a node to NHWC even if it is not claimed by Systolic
  // This is used for the pre-training NHWC transform pass
  bool pretraining_pass_;

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

bool IsAttributeUnsetOrWithExpectedValue(const Node& node, const std::string& attr_name, const std::string& expected_value) {
  const auto* attr_proto = graph_utils::GetNodeAttribute(node, attr_name);
  if ((nullptr != attr_proto) && attr_proto->has_s()) {
    return attr_proto->s() == expected_value;
  }
  return true;
}

bool IsAttributeUnsetOrWithExpectedValue(const Node& node, const std::string& attr_name, int64_t expected_value) {
  const auto* attr_proto = graph_utils::GetNodeAttribute(node, attr_name);
  if ((nullptr != attr_proto) && attr_proto->has_i()) {
    return attr_proto->i() == expected_value;
  }
  return true;
}

bool IsAttributeUnsetOrWithExpectedValues(const Node& node, const std::string& attr_name, const std::vector<int64_t>& expected_values) {
  const auto* attr_proto = graph_utils::GetNodeAttribute(node, attr_name);
  if ((nullptr == attr_proto) || attr_proto->ints_size() == 0) {
    return true;
  }
  if (attr_proto->ints_size() != (int)expected_values.size()) {
    return false;
  }
  for (int i = 0; i < attr_proto->ints_size(); i++) {
    if (attr_proto->ints(i) != expected_values[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
void OcIcHW_to_HWIO(int OC, int IC, int H, int W, const T* oihw_input, std::vector<T>& hwio_output) {
  for (int k = 0; k < H * W; k++) {
    for (int ic = 0; ic < IC; ic++) {
      for (int oc = 0; oc < OC; oc++) {
        hwio_output[k * IC * OC + ic * OC + oc] = oihw_input[oc * H * W * IC + ic * H * W + k];
      }
    }
  }
}

bool SystolicNhwcTransformerImpl::ShouldTransform(Node& node, const logging::Logger& logger) {
  if (this->pretraining_pass_) {
    LOGS(logger, VERBOSE) << "Converting " << node.OpType() << " to NHWC since pretraining_pass is set";
    return true;
  } else if (node.GetExecutionProviderType() == kSystolicExecutionProvider) {
    LOGS(logger, VERBOSE) << "Converting " << node.OpType() << " to NHWC since it is claimed by Systolic";
    return true;
  }
  LOGS(logger, VERBOSE) << "Not converting " << node.OpType() << " to NHWC; not claimed by Systolic and pretraining_pass=false";
  return false;
}

size_t SystolicNhwcTransformerImpl::RemoveOutputEdges(Node& node, int idx) {
  size_t output_edges_count = graph_utils::RemoveNodeOutputEdges(graph_, node, idx);
  // Bias the edge count to handle the case of a node that produces a graph
  // output.
  auto GraphOutputIdxs = graph_.GetNodeOutputsInGraphOutputs(node);
  if (std::find(GraphOutputIdxs.begin(), GraphOutputIdxs.end(), idx) != GraphOutputIdxs.end()) {
    output_edges_count++;
  }
  return output_edges_count;
}

void SystolicNhwcTransformerImpl::CreateNhwcArgument(Node& node,
                                                     Node& nhwc_node, const std::string& basename) {
  auto& output_defs = nhwc_node.MutableOutputDefs();
  for (int idx = 0; idx < (int)output_defs.size(); idx++) {
    size_t original_uses = RemoveOutputEdges(node, idx);

    // Create a new NodeArg to track the output from the NHWC node.
    auto* output_original_arg = output_defs[idx];
    std::string output_reorder_def_name = graph_.GenerateNodeArgName(basename + "_nhwc_" + std::to_string(idx));
    auto* output_nhwc_arg = &graph_.GetOrCreateNodeArg(output_reorder_def_name, nullptr);
    nhwc_args_[output_original_arg] =
        onnxruntime::make_unique<NhwcArgument>(nhwc_node, output_nhwc_arg, original_uses);
    output_defs[idx] = output_nhwc_arg;
  }
}

void SystolicNhwcTransformerImpl::FuseNhwcArgument(Node& node, const NhwcArgument& nhwc_arg) {
  size_t original_uses = RemoveOutputEdges(node, /*idx=*/0);

  // Associate the existing NHWC NodeArg with the output from this node.
  auto* output_original_arg = node.MutableOutputDefs()[0];
  auto& nhwc_node = nhwc_arg.output_node_;
  auto* output_nhwc_arg = nhwc_node.MutableOutputDefs()[0];
  nhwc_args_[output_original_arg] =
      onnxruntime::make_unique<NhwcArgument>(nhwc_node, output_nhwc_arg, original_uses);
}

void SystolicNhwcTransformerImpl::InsertReorderInput(Node& node) {
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
    reorder_input_node.AddAttribute("perm", std::vector<int64_t>({0, 2, 3, 1}));
    reorder_input_node.SetExecutionProviderType(kCpuExecutionProvider);
    input_defs[0] = input_nhwc_arg;
  } else {
    input_defs[0] = it->second;
  }
}

/**
 * Transform conv to NHWC layout, taking care to update the weights to HWIO
 * T parameter is C++ data type of the weight
 * TT parameter is the ONNX Tensor Proto DataType
 */
template <typename T, ONNX_NAMESPACE::TensorProto_DataType ONNX_T>
void SystolicNhwcTransformerImpl::TransformConv(Node& node, const logging::Logger& logger, uint32_t weightIdx, uint32_t biasIdx) {
  if (!ShouldTransform(node, logger)) {
    return;
  }
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // Require that the weights tensor be static, and has exactly 4 dims
  const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph_, *input_defs[weightIdx]) ||
      !graph_.GetInitializedTensor(input_defs[weightIdx]->Name(), conv_W_tensor_proto) ||
      (conv_W_tensor_proto->data_type() != ONNX_T) ||
      (conv_W_tensor_proto->dims_size() != 4)) {
    LOGS(logger, VERBOSE) << "Weights tensor for " << node.OpType() << " not static. Bailing.";
    return;
  }

  // Unused, but keeping just for reference
  // const int64_t output_channels = conv_W_tensor_proto->dims(0);
  // const int64_t input_channels = conv_W_tensor_proto->dims(1);
  // const int64_t kernel_size = conv_W_tensor_proto->dims(2) * conv_W_tensor_proto->dims(3);
  // const int64_t kernel_dim = input_channels * kernel_size;

  // int64_t group_count;
  // const auto* group_attr = graph_utils::GetNodeAttribute(node, "group");
  // if (group_attr != nullptr && utils::HasInt(*group_attr)) {
  //   group_count = group_attr->i();
  // } else {
  //   group_count = 1;
  // }

  NodeArg* nhwc_conv_W_arg;
  auto filters_it = filters_transposed.find(input_defs[weightIdx]);
  if (filters_it != filters_transposed.end()) {
    // Reuse the existing NodeArg.
    nhwc_conv_W_arg = filters_it->second;
  } else {
    Initializer conv_W{*conv_W_tensor_proto, graph_.ModelPath()};
    std::vector<T> reordered_filter(conv_W.size());

    // We convert from OcIcHW format to HWIO format
    int OC = conv_W.dims()[0];
    int IC = conv_W.dims()[1];
    int H = conv_W.dims()[2];
    int W = conv_W.dims()[3];

    OcIcHW_to_HWIO(OC, IC, H, W, conv_W.data<T>(), reordered_filter);

    ONNX_NAMESPACE::TensorProto nhwc_conv_W_tensor_proto;
    nhwc_conv_W_tensor_proto.set_data_type(ONNX_T);
    nhwc_conv_W_tensor_proto.set_name(graph_.GenerateNodeArgName(input_defs[weightIdx]->Name() + "_nhwc"));
    nhwc_conv_W_tensor_proto.set_raw_data(reordered_filter.data(), reordered_filter.size() * sizeof(T));

    nhwc_conv_W_tensor_proto.add_dims(H);
    nhwc_conv_W_tensor_proto.add_dims(W);
    nhwc_conv_W_tensor_proto.add_dims(IC);
    nhwc_conv_W_tensor_proto.add_dims(OC);

    nhwc_conv_W_arg = &graph_utils::AddInitializer(graph_, nhwc_conv_W_tensor_proto);
    filters_transposed.emplace(input_defs[weightIdx], nhwc_conv_W_arg);
  }

  // Create the replacement node.
  std::string nhwc_node_name = graph_.GenerateNodeName(node.Name() + "_nhwc");
  Node& nhwc_node = graph_.AddNode(nhwc_node_name,
                                   node.OpType() + "_nhwc",
                                   nhwc_node_name,
                                   input_defs,
                                   output_defs,
                                   &node.GetAttributes(),
                                   kOnnxDomain);
  nhwc_node.SetExecutionProviderType(node.GetExecutionProviderType());

  nhwc_node.MutableInputDefs()[weightIdx] = nhwc_conv_W_arg;

  if (input_defs.size() == biasIdx + 1) {
    nhwc_node.MutableInputDefs()[biasIdx] = input_defs[biasIdx];
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

  LOGS(logger, VERBOSE) << "Transforming " << node.OpType() << " to NHWC";
  CreateNhwcArgument(node, nhwc_node, output_defs[0]->Name());
  removed_nodes_.push_front(node.Index());
}

/**
 * Transform for nodes that don't care about input format. E.g. relu
 */
void SystolicNhwcTransformerImpl::TransformPassThrough(Node& node, const logging::Logger& logger) {
  if (!ShouldTransform(node, logger)) {
    return;
  }
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  auto it = nhwc_args_.find(input_defs[0]);
  // If the input is indeed in NHWC format
  if (it != nhwc_args_.end()) {
    LOGS(logger, VERBOSE) << "Transforming pass through node " << node.OpType() << " to NHWC";
    auto& nhwc_input = it->second;
    input_defs[0] = nhwc_input->nhwc_arg_;
    nhwc_input->remaining_original_uses_--;

    CreateNhwcArgument(node, node, output_defs[0]->Name());
  }
}

void SystolicNhwcTransformerImpl::TransformQLinearAdd(Node& node, const logging::Logger& logger) {
  if (!ShouldTransform(node, logger)) {
    return;
  }
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  NodeArg* tensor1 = input_defs[0];
  NodeArg* tensor2 = input_defs[3];  // N.b. [3] since we scale/zp is other two
  auto it = nhwc_args_.find(tensor1);
  auto it2 = nhwc_args_.find(tensor2);

  // If both inputs are indeed in NHWC format
  if (it != nhwc_args_.end() && it2 != nhwc_args_.end()) {
    // Note that they must be in the same shape (no broadcasting)
    if (tensor1->Shape() == nullptr ||
        tensor2->Shape() == nullptr) {
      LOGS(logger, VERBOSE) << "No shape inference data for QLinearAdd. Bailing on NHWC.";
      return;
    }

    TensorShape shape1 = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*tensor1->Shape());
    TensorShape shape2 = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*tensor1->Shape());

    // If shapes don't match or unknown size, can't do NHWC conversion
    if (shape1 != shape2 || shape1.Size() == -1) {
      LOGS(logger, VERBOSE) << "Shapes do not match. Bailing on NHWC.";
      LOGS(logger, VERBOSE) << "T1 shape:" << shape1.ToString() << " T2 shape: " << shape2.ToString();
    }

    auto& nhwc_input1 = it->second;
    auto& nhwc_input2 = it2->second;

    LOGS(logger, VERBOSE) << "Transforming QLinearAdd to NHWC";

    input_defs[0] = nhwc_input1->nhwc_arg_;
    input_defs[3] = nhwc_input2->nhwc_arg_;
    nhwc_input1->remaining_original_uses_--;
    nhwc_input2->remaining_original_uses_--;

    CreateNhwcArgument(node, node, output_defs[0]->Name());
  }
}

bool SystolicNhwcTransformerImpl::FuseMaxPoolWithConv(Node& node, const logging::Logger& logger) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // MaxPool has a variant that returns indices as well... we can't support that
  if (output_defs.size() != 1) {
    return false;
  }

  auto it = nhwc_args_.find(input_defs[0]);
  // If the input is indeed in NHWC format
  if (it == nhwc_args_.end()) {
    return false;
  }

  auto& nhwc_input = it->second;
  // Check whether the input to this node is a QLinearConv node and the output of QLinearConv only goes to this node
  if (nhwc_input->output_node_.OpType() != "QLinearConv_nhwc" && nhwc_input->starting_original_uses_ != 1) {
    return false;
  }

  input_defs[0] = nhwc_input->nhwc_arg_;
  nhwc_input->remaining_original_uses_--;

  LOGS(logger, VERBOSE) << "Trying to fuse MaxPool (contingent on attrs)";

  // We only handle the most common cases (namely no dilations). Systolic has its own
  // limitations which we check before dispatching to it (otherwise we handle it on cpu).
  std::vector<int64_t> kernel_shape;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(node, "kernel_shape", kernel_shape) && kernel_shape.size() == 2)) {
    return false;
  }
  if (!IsAttributeUnsetOrWithExpectedValue(node, "auto_pad", "NOTSET")) {
    return false;
  }
  if (!IsAttributeUnsetOrWithExpectedValue(node, "ceil_mode", 0)) {
    return false;
  }
  if (!IsAttributeUnsetOrWithExpectedValue(node, "dilations", {1, 1})) {
    return false;
  }
  if (!IsAttributeUnsetOrWithExpectedValue(node, "storage_order", 0)) {
    return false;
  }

  LOGS(logger, VERBOSE) << "We can handle these params. Fusing MaxPool with preceding NHWC conv";

  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> strides = {1, 1};
  graph_utils::GetRepeatedNodeAttributeValues(node, "pads", pads);
  graph_utils::GetRepeatedNodeAttributeValues(node, "strides", strides);

  nhwc_input->output_node_.AddAttribute("maxpool", static_cast<int64_t>(1));
  nhwc_input->output_node_.AddAttribute("pool_kernel_shape", kernel_shape);
  nhwc_input->output_node_.AddAttribute("pool_pads", pads);
  nhwc_input->output_node_.AddAttribute("pool_strides", strides);

  // Reset the shape of this to avoid errors during subsequent shape inference
  output_defs[0]->ClearShape();
  FuseNhwcArgument(node, *nhwc_input);
  removed_nodes_.push_front(node.Index());
  return true;
}

void SystolicNhwcTransformerImpl::TransformMaxPool(Node& node, const logging::Logger& logger) {
  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  // If this isn't the pre-training pass we can try to fuse a maxpool with a conv
  if (!this->pretraining_pass_ && FuseMaxPoolWithConv(node, logger)) {
    return;
  }

  // However if the fusino is unsuccessful (or if we're in pre-trianing) then fall-back to separate NHWC Pool node
  auto it = nhwc_args_.find(input_defs[0]);
  // If the input is indeed in NHWC format
  if (it == nhwc_args_.end()) {
    return;
  }

  if (!IsAttributeUnsetOrWithExpectedValue(node, "storage_order", 0)) {
    return;
  }
  if (!IsAttributeUnsetOrWithExpectedValue(node, "ceil_mode", 0)) {
    return;
  }
  // Create the replacement node.
  std::string nhwc_node_name = graph_.GenerateNodeName(node.Name() + "_nhwc");
  Node& nhwc_node = graph_.AddNode(nhwc_node_name,
                                   node.OpType() + "_nhwc",
                                   nhwc_node_name,
                                   input_defs,
                                   output_defs,
                                   &node.GetAttributes(),
                                   kOnnxDomain);
  nhwc_node.SetExecutionProviderType(node.GetExecutionProviderType());
  printf("MaxPool Output Edges %zd\n", node.GetOutputEdgesCount());

  auto* nhwc_input = it->second.get();
  nhwc_node.MutableInputDefs()[0] = nhwc_input->nhwc_arg_;
  nhwc_input->remaining_original_uses_--;

  LOGS(logger, VERBOSE) << "Transforming " << node.OpType() << " to NHWC";
  CreateNhwcArgument(node, nhwc_node, output_defs[0]->Name());
  removed_nodes_.push_front(node.Index());
}

void SystolicNhwcTransformerImpl::Transform(Node& node, const logging::Logger& logger) {
  if (node.OpType() == "QLinearConv") {
    TransformConv<int8_t, ONNX_NAMESPACE::TensorProto_DataType_INT8>(node, logger, /*weightIdx= */ 3, /*biasIdx= */ 8);
  } else if (node.OpType() == "Conv") {
    TransformConv<float, ONNX_NAMESPACE::TensorProto_DataType_FLOAT>(node, logger, /*weightIdx= */ 1, /*biasIdx= */ 2);
  } else if (node.GetInputEdgesCount() == 0 && node.InputDefs().size() != 0) {
    // The following transforms only run when the input edge count has already
    // been decremented to zero by earlier transforms. This is a hint that the
    // node may already have all inputs converted to NHWC format and is not
    // needed for correct operation. This avoids doing extra string checks for
    // nodes unrelated to this transformer.
    if (node.OpType() == "QLinearRelu" || node.OpType() == "Relu") {
      TransformPassThrough(node, logger);
    } else if (node.OpType() == "QLinearAdd") {
      TransformQLinearAdd(node, logger);
    } else if (node.OpType() == "MaxPool") {
      TransformMaxPool(node, logger);
    }
  }

  // The node may not match any of the checks above or may not have been
  // transformed for other reasons such as unsupported attributes or alignment.
  // However, the node may still use an input that has been produced by a NHWC
  // node. Finalize() walks through the list of NHWC outputs and inserts the
  // needed reorder operations to ensure that these inputs remain in NCHW
  // format.
}

void SystolicNhwcTransformerImpl::Finalize(bool& modified, const logging::Logger& logger) {
  // Create ReorderOutput nodes for any NHWC outputs that still have uses with
  // the original tensor format.
  for (auto& nhwc_output : nhwc_args_) {
    if (nhwc_output.second->remaining_original_uses_ > 0) {
      auto* output_original_arg = nhwc_output.first;
      auto* output_nhwc_arg = nhwc_output.second->nhwc_arg_;
      LOGS(logger, VERBOSE) << "Inserting reorder to NCHW from " << output_nhwc_arg->Name() << " to " << output_original_arg->Name();
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

Status SystolicNhwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  if (this->pretraining_pass_) {
    LOGS(logger, VERBOSE) << "All potential nodes will be converted to NHWC. This should only be done pre-training";
  }
  SystolicNhwcTransformerImpl impl(graph, this->pretraining_pass_);
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    impl.Transform(node, logger);
  }
  impl.Finalize(modified, logger);
  return Status::OK();
}

}  // namespace onnxruntime
