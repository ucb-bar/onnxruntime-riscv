// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <iostream>
#include <vector>
#include <systolic/systolic_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#ifdef USE_CUSTOM_OP_LIBRARY
#include "custom_op_library.h"
#endif
#ifdef FOR_FIRESIM
#include <sys/mman.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
  session_options.SetIntraOpNumThreads(1);
  if (cmd.count("trace")) {
    session_options.EnableProfiling(cmd["trace"].as<std::string>().c_str());
  }
  
  printf("Using systolic in mode %d\n", cmd["execution"].as<int>());

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Systolic(session_options, /*use_arena=*/ 1, /*accelerator_mode=*/ (char) cmd["execution"].as<int>()));

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

  //*************************************************************************
  // create session and load model into memory
  // using squeezenet version 1.3
  // URL = https://github.com/onnx/models/tree/master/squeezenet

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
  
  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values


  printf("Loading image\n");

  int dimX, dimY, numChannels;
  unsigned char *data = stbi_load(cmd["image"].as<std::string>().c_str(), &dimX, &dimY, &numChannels, 0);

  if (data == nullptr) {
    printf("Could not load image\n");
    return -1;
  }
  printf("Image dimensions: %d %d %d\n", dimX, dimY, numChannels);
  if (numChannels != 3) {
    printf("Loaded image has more than 3 channels. Use JPG instead of PNG\n");
    return -1;
  }

  // PYTORCH diff. If you are using model exported from pytorch with a fixed dimension
  // comment out the following two assignments
  input_node_dims[1] = dimY;
  input_node_dims[2] = dimX;

  size_t input_tensor_size = 3 * dimX * dimY;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!  
  float *input_tensor_values = new float[input_tensor_size];

  
  for (int i = 0; i < dimY; i++) {
    for (int j = 0; j < dimX; j++) {
      unsigned char r = *(data++);
      unsigned char g = *(data++);
      unsigned char b = *(data++);

      input_tensor_values[(0*dimY + i)*dimX + j] = b - 102.9801;
      input_tensor_values[(1*dimY + i)*dimX + j] = g - 115.9465;
      input_tensor_values[(2*dimY + i)*dimX + j] = r - 122.7717;

      // If you are using model from pytorch, use below preprocess
      // input_tensor_values[(0*dimY + i)*dimX + j] = (r/255.0 - 0.485)/0.229;
      // input_tensor_values[(1*dimY + i)*dimX + j] = (g/255.0 - 0.456)/0.224;
      // input_tensor_values[(2*dimY + i)*dimX + j] = (b/255.0 - 0.406)/0.225;  
    }
  }
  printf("First few image values %f %f %f\n", input_tensor_values[0], input_tensor_values[1], input_tensor_values[2]);

  // initialize input data with values in [0.0, 1.0]
  // for (unsigned int i = 0; i < input_tensor_size; i++)
  //   input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values, input_tensor_size, input_node_dims.data(), input_node_dims.size());
  assert(input_tensor.IsTensor());

  printf("Starting inference\n");

  auto pre_inference_cycles = read_cycles();

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());
  auto post_inference_cycles = read_cycles();

  printf("Number of output tensors %lld\n", output_tensors.size());

  for (int i = 0; i < output_tensors.size(); i++) {
    printf("Dimensions\n");
    std::vector<int64_t> output_node_dims = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
    printf("num_dims=%zu: [", output_node_dims.size());
    for (size_t j = 0; j < output_node_dims.size(); j++) {
      printf("%jd, ", output_node_dims[j]);
    }
    printf("]\n");
  }

  float *boxes = output_tensors[0].GetTensorMutableData<float>();
  int64_t *labels = output_tensors[1].GetTensorMutableData<int64_t>();
  float *scores = output_tensors[2].GetTensorMutableData<float>();
  float *masks = output_tensors[3].GetTensorMutableData<float>();


  FILE *f = fopen("boxes.data", "wb");
  fwrite(boxes, sizeof(float), output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(), f);
  fclose(f);

  f = fopen("labels.data", "wb");
  fwrite(labels, sizeof(int64_t), output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount(), f);
  fclose(f);

  f = fopen("scores.data", "wb");
  fwrite(scores, sizeof(float), output_tensors[2].GetTensorTypeAndShapeInfo().GetElementCount(), f);
  fclose(f);

  f = fopen("masks.data", "wb");
  fwrite(masks, sizeof(float), output_tensors[3].GetTensorTypeAndShapeInfo().GetElementCount(), f);
  fclose(f);



  // assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // // Get pointer to output tensor float values
  // float* floatarr = output_tensors.front().GetTensorMutableData<float>();

  // printf("Element count %ld. Top 5 classes:\n", output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount());
  // auto topK = getTopK(floatarr, output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount(), 5);
  // while (!topK.empty()) {
  //   std::pair<float, int> val = topK.top();
  //   printf("%f %s\n", val.first, imagenet_labels[val.second]);
  //   topK.pop();
  // }

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
  return 0;
}
