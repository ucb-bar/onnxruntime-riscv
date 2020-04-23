// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <systolic/systolic_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#ifdef USE_CUSTOM_OP_LIBRARY
#include "custom_op_library.h"
#endif

#include "tensor_helper.h"
#include "cmd_args.h"

unsigned long long read_cycles()
{
    unsigned long long cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);
  printf("Loaded runner program\n");

  cxxopts::ParseResult cmd = parse(argc, argv);

  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(static_cast<OrtLoggingLevel>(cmd["debug"].as<int>()), "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  if (cmd.count("trace")) {
    session_options.EnableProfiling(cmd["trace"].as<std::string>().c_str());
  }
  
  printf("Using systolic in mode %d\n", cmd["execution"].as<int>());

  // Use systolic array execution provider
  Ort::ThrowOnError(
    OrtSessionOptionsAppendExecutionProvider_Systolic(
      session_options,
      /* use_arena = */ 1,
      /* accelerator_mode = */ (char) cmd["execution"].as<int>()
    )
  );

  // Sets graph optimization level
  // Available levels are
  // 0: ORT_DISABLE_ALL -> To disable all optimizations
  // 1: ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // 2: ORT_ENABLE_EXTENDED -> To enable extended optimizations
  //    (Includes level 1 + more complex optimizations like node fusions)
  // 99: ORT_ENABLE_ALL -> To Enable All possible opitmizations
  int optimization_level = cmd["optimization"].as<int>();
  session_options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(optimization_level));

  if (cmd.count("save_model")) {
    session_options.SetOptimizedModelFilePath(cmd["save_model"].as<std::string>().c_str());
  }

#ifdef USE_CUSTOM_OP_LIBRARY
  if (cmd.count("kernel")) {
    printf("Loading custom kernel\n");
    Ort::ThrowOnError(RegisterCustomOps((OrtSessionOptions*) session_options, OrtGetApiBase()));
  }
#endif

  // create session and load model into memory
  // using BERT Squad
  // URL = https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad

  const char* model_path = cmd["model"].as<std::string>().c_str();

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<size_t> input_node_elem_count;

  printf("Number of inputs = %zu\n", num_input_nodes);
  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %ld : name=%s, ", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("type=%d, ", type);

    // print input shapes/dims
    input_node_dims.push_back(tensor_info.GetShape());
    printf("num_dims=%zu: [", input_node_dims[i].size());
    for (size_t j = 0; j < input_node_dims[i].size(); j++) {
      printf("%jd, ", input_node_dims[i][j]);
    }
    printf("]\n");
  }

  if (num_input_nodes != 4) {
    printf("ERROR: Graph has the wrong number of input nodes defined.\n");
    return -1;
  }

  // Results should be...
  // Number of inputs = 4

  size_t num_output_nodes = session.GetOutputCount();
  printf("Number of outputs = %zu\n", num_output_nodes);
  std::vector<const char*> output_node_names(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; i++) {
    // print input node names
    char* output_name = session.GetOutputName(i, allocator);
    printf("Output %ld : name=%s, ", i, output_name);
    output_node_names[i] = output_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("type=%d, ", type);

    // print input shapes/dims
    std::vector<int64_t> output_node_dims = tensor_info.GetShape();
    printf("num_dims=%zu: [", output_node_dims.size());
    for (size_t j = 0; j < output_node_dims.size(); j++) {
      // just for accurately printing the dimensions
      printf("%jd, ", output_node_dims[j]);
    }
    printf("]\n");
  }

  if (output_node_names.size() != 3) {
    printf("ERROR: Graph has the wrong number of output nodes defined.\n");
    return -1;
  }

  if (input_node_dims[0].size() != 1 || input_node_dims[1].size() != 2
      || input_node_dims[2].size() != 2 || input_node_dims[3].size() != 2) {
    std::cout << "Wrong number of input dimensions in at least one case!" << std::endl;
  }
  
  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  //size_t input_tensor_size = 3 * 224 * 224;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<long> unique_ids_values;
  std::vector<long> input_ids_values;
  std::vector<long> input_masks_values;
  std::vector<long> segment_ids_values;

  // Read in the dimensions before anything else
  long dims[2];

  std::ifstream fs;
  fs.open("input_data/dimensions.bin", std::ios::in | std::ios::binary);
  fs.read((char *)&dims, sizeof(long) * 2);
  fs.close();
  fs.clear();

  printf("Dimensions are: (%ld, %ld)\n", dims[0], dims[1]);
  if (dims[1] != 256) {
    printf("Wrong second dimension of bert input: require 256, found %ld\n", dims[1]);
    return -1;
  }
  // Replace -1 dimension sizes with dynamic value
  input_node_dims[0][0] = dims[0];
  input_node_dims[1][0] = dims[0];
  input_node_dims[2][0] = dims[0];
  input_node_dims[3][0] = dims[0];
  
  // Read in the input tensors from the preprocessor script
  unique_ids_values.resize(dims[0]);
  input_ids_values.resize(dims[0] * dims[1]);
  input_masks_values.resize(dims[0] * dims[1]);
  segment_ids_values.resize(dims[0] * dims[1]);

  fs.open("input_data/unique_ids.bin", std::ios::in | std::ios::binary);
  fs.read((char *)unique_ids_values.data(), sizeof(long) * dims[0]);
  fs.close();
  fs.clear();
  printf("The unique ids are: %ld, %ld, %ld\n",
      unique_ids_values[0], unique_ids_values[1], unique_ids_values[2]);

  fs.open("input_data/input_ids.bin", std::ios::in | std::ios::binary);
  fs.read((char *)input_ids_values.data(), sizeof(long) * dims[0] * dims[1]);
  fs.close();
  fs.clear();
  printf("The first few input ids are: %ld, %ld, %ld\n",
      input_ids_values[0], input_ids_values[1], input_ids_values[2]);

  fs.open("input_data/input_masks.bin", std::ios::in | std::ios::binary);
  fs.read((char *)input_masks_values.data(), sizeof(long) * dims[0] * dims[1]);
  fs.close();
  fs.clear();
  printf("The first few input masks are: %ld, %ld, %ld\n",
      input_masks_values[0], input_masks_values[1], input_masks_values[2]);

  fs.open("input_data/segment_ids_values.bin", std::ios::in | std::ios::binary);
  fs.read((char *)input_masks_values.data(), sizeof(long) * dims[0] * dims[1]);
  fs.close();
  fs.clear();
  printf("The first few segment ids are: %ld, %ld, %ld\n",
      segment_ids_values[0], segment_ids_values[1], segment_ids_values[2]);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<Ort::Value> bert_inputs;
  bert_inputs.push_back(Ort::Value::CreateTensor<long>(memory_info,
                                                       unique_ids_values.data(), dims[0],
                                                       input_node_dims[0].data(), 1));
  bert_inputs.push_back(Ort::Value::CreateTensor<long>(memory_info,
                                                       input_ids_values.data(), dims[0] * dims[1],
                                                       input_node_dims[1].data(), 2));
  bert_inputs.push_back(Ort::Value::CreateTensor<long>(memory_info,
                                                       input_masks_values.data(), dims[0] * dims[1],
                                                       input_node_dims[2].data(), 2));
  bert_inputs.push_back(Ort::Value::CreateTensor<long>(memory_info,
                                                       segment_ids_values.data(), dims[0] * dims[1],
                                                       input_node_dims[3].data(), 2));


  auto pre_inference_cycles = read_cycles();

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                    input_node_names.data(),
                                    bert_inputs.data(), num_input_nodes,
                                    output_node_names.data(), num_output_nodes);

  auto post_inference_cycles = read_cycles();
  printf("Done! Inference took %llu cycles \n", (post_inference_cycles - pre_inference_cycles));

  assert(output_tensors.size() == 3);

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();

  printf("Element count %ld. Top 5 classes:\n", output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount());
  auto topK = getTopK(floatarr, output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount(), 5);
  while (!topK.empty()) {
    std::pair<float, int> val = topK.top();
    //printf("%f %s\n", val.first, imagenet_labels[val.second]);
    topK.pop();
  }

  // score the model, and print scores for first 5 classes
  // for (int i = 0; i < 5; i++)
  //   printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317
  return 0;
}
