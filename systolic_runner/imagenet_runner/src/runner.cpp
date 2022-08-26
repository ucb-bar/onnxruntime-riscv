// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <stdlib.h>
#include <string>

#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <systolic/systolic_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#ifdef FOR_FIRESIM
#include <sys/mman.h>
#endif

#ifdef USE_CUSTOM_OP_LIBRARY
#include "custom_op_library.h"
#endif
#ifdef USE_HWACHA
#include <hwacha/hwacha_provider_factory.h>
#endif

#include "stb_image.h"

#include "tensor_helper.h"
#include "cmd_args.h"
#include "labels.h"

#include <thread>

bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int getLabelOfBatchImage(const std::string &path) {
  size_t lastidx = path.find_last_of("/\\");
  size_t secondlastidx = path.find_last_of("/\\", lastidx - 1);
  return std::stoi(path.substr(secondlastidx + 1, (lastidx - secondlastidx - 1)));
}


const char* imagenet_labels[1000] = IMAGENET_LABELS;

unsigned long long read_cycles()
{
    unsigned long long cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

std::vector<int> inferOnImage(const std::string &path, const std::string &preprocess, Ort::Session &session,
                  const std::vector<const char*> &input_node_names,
                  const std::vector<int64_t> &input_node_dims,
                  const std::vector<const char*> &output_node_names) {
  size_t input_tensor_size = 3 * 224 * 224;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!
  printf("Loading image\n");

  int dimX, dimY, numChannels;
  unsigned char *data = stbi_load(path.c_str(), &dimX, &dimY, &numChannels, 0);
  unsigned char *orig_data = data;

  if (data == nullptr) {
    printf("Could not load image\n");
    exit(-1);
  }
  printf("Image dimensions: %d %d %d\n", dimX, dimY, numChannels);
  if (numChannels != 3) {
    printf("Loaded image has more than 3 channels. Use JPG instead of PNG\n");
    exit(-1);
  }

  
  std::vector<float> input_tensor_values(input_tensor_size);

  
  for (int i = 0; i < 224; i++) {
    for (int j = 0; j < 224; j++) {
      unsigned char r = *(data++);
      unsigned char g = *(data++);
      unsigned char b = *(data++);

      if (preprocess == "caffe2") {
        input_tensor_values[(0*224 + i)*224 + j] = b - 103.939;
        input_tensor_values[(1*224 + i)*224 + j] = g - 116.779;
        input_tensor_values[(2*224 + i)*224 + j] = r - 123.68;  
      } 
      else if (preprocess == "caffe") {
        input_tensor_values[(0*224 + i)*224 + j] = (b - 103.94)*0.017;
        input_tensor_values[(1*224 + i)*224 + j] = (g - 116.78)*0.017;
        input_tensor_values[(2*224 + i)*224 + j] = (r - 123.68)*0.017;  
      } else if (preprocess == "mxnet") {
        input_tensor_values[(0*224 + i)*224 + j] = (b/255.0 - 0.406)/0.225;
        input_tensor_values[(1*224 + i)*224 + j] = (g/255.0 - 0.456)/0.224;
        input_tensor_values[(2*224 + i)*224 + j] = (r/255.0 - 0.485)/0.229;  
      } else {
        std::cout << "Unknown preprocess option: " << preprocess << std::endl;
        exit(1);
      }
    }
  }
  stbi_image_free(orig_data);
  printf("First few image values %f %f %f\n", input_tensor_values[0], input_tensor_values[1], input_tensor_values[2]);

  // initialize input data with values in [0.0, 1.0]
  // for (unsigned int i = 0; i < input_tensor_size; i++)
  //   input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  auto pre_inference_cycles = read_cycles();

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  auto post_inference_cycles = read_cycles();

  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();

  printf("Element count %ld. Top 5 classes:\n", output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount());
  auto topK = getTopK(floatarr, output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount(), 5);
  std::vector<int> topFive; topFive.reserve(5);
  while (!topK.empty()) {
    std::pair<float, int> val = topK.top();
    topFive.push_back(val.second);
    assert(val.second < 1000 && "Returned label out of bounds");
    printf("%f %s\n", val.first, imagenet_labels[val.second]);
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
  printf("Done! Inference took %llu cycles \n", (post_inference_cycles - pre_inference_cycles));
  return topFive;
}

//#include <sys/mman.h> 

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);
  printf("Loaded runner program\n");

  // Use for firesim
#ifdef FOR_FIRESIM
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
  } else {
    printf("Finished mlockall\n");
  }
#endif

  cxxopts::ParseResult cmd = parse(argc, argv);
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(static_cast<OrtLoggingLevel>(cmd["debug"].as<int>()), "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  int nthreads = std::thread::hardware_concurrency();
  char *nthreads_setting = std::getenv("NTHREADS");

  printf("nthreads_setting: %s\n", nthreads_setting);

  if (nthreads_setting != NULL) {
    try {
      nthreads = std::stoi(nthreads_setting);
    }
    catch(std::invalid_argument const& ex) { }
  }

  printf("nthreads: %d\n", nthreads);
  printf("getenv test: %s\n", std::getenv("PATH"));

  session_options.SetIntraOpNumThreads(nthreads);
  
  if (cmd.count("trace")) {
    session_options.EnableProfiling(cmd["trace"].as<std::string>().c_str());
  }
  
  printf("Using systolic in mode %d\n", cmd["execution"].as<int>());
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Systolic(session_options, /*use_arena=*/ 1, /*accelerator_mode=*/ (char) cmd["execution"].as<int>()));
#ifdef USE_HWACHA
  printf("Using hwacha\n");
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Hwacha(session_options, /*use_arena=*/ 1));
#endif

  // Sets graph optimization level
  // Available levels are
  // 0: ORT_DISABLE_ALL -> To disable all optimizations
  // 1: ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // 2: ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // 99: ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(cmd["optimization"].as<int>()));

  if (cmd.count("save_model")) {
    session_options.SetOptimizedModelFilePath(cmd["save_model"].as<std::string>().c_str());
  }

#ifdef USE_CUSTOM_OP_LIBRARY
  if (cmd.count("kernel")) {
    printf("Loading custom kernel\n");
    Ort::ThrowOnError(RegisterCustomOps((OrtSessionOptions*) session_options, OrtGetApiBase()));
  }
#endif

  const char* model_path = cmd["model"].as<std::string>().c_str();

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

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
    input_node_dims = tensor_info.GetShape();
    printf("num_dims=%zu: [", input_node_dims.size());
    for (size_t j = 0; j < input_node_dims.size(); j++) {
      printf("%jd, ", input_node_dims[j]);
    }
    printf("]\n");
  }

  if (num_input_nodes > 1) {
    printf("ERROR: Graph has multiple input nodes defined.\n");
    return -1;
  }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

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
      printf("%jd, ", output_node_dims[j]);
    }
    printf("]\n");
  }

  if (output_node_names.size() > 1) {
    printf("ERROR: Graph has multiple output nodes defined. Please specify an output manually.\n");
    return -1;
  }
  
  if (has_suffix(cmd["image"].as<std::string>(), ".txt"))
  {
    int topFiveRight = 0;
    int topOneRight = 0;
    int totalCases = 0;
    // Batch inference
    std::ifstream batch_in(cmd["image"].as<std::string>());
    std::string path;

    int startLine = 1;
    int endLine = -1;

    if (cmd.count("range")) {
      std::vector<int> startEndPair = cmd["range"].as<std::vector<int>>();
      assert(startEndPair.size() >= 1 && startEndPair.size() <= 2 && "Expected range in form of 'start, end'");
      startLine = startEndPair[0];
      if (startEndPair.size() == 2) {
        endLine = startEndPair[1];
      }
    }

    printf("Batch processing lines %d - %d\n", startLine, endLine);

    int curLine = 0;
    while (std::getline(batch_in, path)) {
      if (path.empty()) continue;
      curLine += 1;
      if (curLine < startLine || (endLine != -1 && curLine > endLine)) {
        continue;
      }

      totalCases += 1;
      // The most likely prediction is at the end of the list
      std::vector<int> topFiveOut = inferOnImage(path, cmd["preprocess"].as<std::string>(),
                  session, input_node_names, input_node_dims, output_node_names);
      int expected_label = getLabelOfBatchImage(path);
      assert(expected_label < 1000 && "Expected label out of bounds");
      printf("Expected was %d - %s\n", expected_label, imagenet_labels[expected_label]);

      for (size_t i = 0; i < topFiveOut.size(); i++) {
        if (topFiveOut[i] == expected_label) {
          topFiveRight += 1;
          topOneRight += ((i + 1) == topFiveOut.size());
          break;
        }
      }

      if (curLine % 100 == 0) {
        printf("Checkpoint! Processed up to line %d\n", curLine);
        printf("Top five right: %d/%d\n", topFiveRight, totalCases);
        printf("Top one right: %d/%d\n", topOneRight, totalCases);
      }
    }
    
    printf("Finished batch\n");
    printf("Top five right: %d/%d\n", topFiveRight, totalCases);
    printf("Top one right: %d/%d\n", topOneRight, totalCases);
  } else {
    inferOnImage(cmd["image"].as<std::string>(), cmd["preprocess"].as<std::string>(),
                  session, input_node_names, input_node_dims, output_node_names);
  }
 
  return 0;
}

