// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

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

#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "mmio.h"

#define AIRSIM_STATUS (ptr + 0x00)
#define AIRSIM_IN     (ptr + 0x08)
#define AIRSIM_OUT    (ptr + 0x0C)

#define CS_GRANT_TOKEN  0x80
#define CS_REQ_CYCLES   0x81
#define CS_RSP_CYCLES   0x82
#define CS_DEFINE_STEP  0x83

#define CS_REQ_WAYPOINT 0x01
#define CS_RSP_WAYPOINT 0x02
#define CS_SEND_IMU     0x03
#define CS_REQ_ARM      0x04
#define CS_REQ_DISARM   0x05
#define CS_REQ_TAKEOFF  0x06

#define CS_REQ_IMG      0x10
#define CS_RSP_IMG      0x11

#define CS_REQ_DEPTH     0x12
#define CS_RSP_DEPTH     0x13

#define CS_SET_TARGETS  0x20

intptr_t ptr;

void send_arm() {
    printf("Sending arm...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_ARM);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

void send_takeoff() {
    printf("Sending takeoff...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_TAKEOFF);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

void send_img_req() {
    printf("Requesting image...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_IMG);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

void send_depth_req() {
    printf("Requesting depth...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_DEPTH);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

float load_depth() {
  uint32_t i;
  uint8_t status;
  uint32_t raw_result;
  float result;

  printf("Receiving depth ...\n");
  do {
    status = reg_read8(AIRSIM_STATUS);
  } while ((status & 0x1) == 0);

  uint32_t cmd = reg_read32(AIRSIM_OUT);
  while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  uint32_t num_bytes = reg_read32(AIRSIM_OUT);
  while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  raw_result = reg_read32(AIRSIM_OUT);
  result = *((float *) &raw_result);
  return result;
}

void load_img_row(unsigned char * buf) {
  uint32_t i;
  uint8_t status;

  // printf("Receiving image ...\n");
  do {
    status = reg_read8(AIRSIM_STATUS);
  } while ((status & 0x1) == 0);

  uint32_t cmd = reg_read32(AIRSIM_OUT);
  // printf("Cmd: %x\n", cmd);
  while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  uint32_t num_bytes = reg_read32(AIRSIM_OUT);
  // printf("Num_bytes: %d\n", num_bytes);
  for(i = 0; i < num_bytes / 4; i++) {
    while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
    ((uint32_t * ) buf)[i] = reg_read32(AIRSIM_OUT);
    // printf("(%d, %x) ", i, buf[i]);
  }
  // printf("\n");

}

void loadSimImage(unsigned char * data, int dimX, int dimY, int numChannels) {
      uint32_t i;

      send_img_req();
      printf("In between cmds...\n");
      for(i = 0; i < dimX; i++) {
        load_img_row(data + dimX * numChannels * i);
      }
}


void send_target(float zcoord, float xvel, float yvel, float yawrate) {
    printf("Setting target %f, %f, %f, %f...\n", zcoord, xvel, yvel, yawrate);

    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_SET_TARGETS);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 16);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &zcoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &xvel));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &yvel));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &yawrate ));
}

void send_waypoint(float xcoord, float ycoord, float zcoord, float vel) {
    printf("Navigating to waypoint...\n",xcoord, ycoord, zcoord);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, CS_RSP_WAYPOINT);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, 16);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, *((uint32_t *) &xcoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, *((uint32_t *) &ycoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, *((uint32_t *) &zcoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, *((uint32_t *) &vel));
}

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


void inferOnImage(unsigned char * data, const std::string &preprocess, Ort::Session &session,
                  const std::vector<const char*> &input_node_names,
                  const std::vector<int64_t> &input_node_dims,
                  const std::vector<const char*> &output_node_names,
                  int dimX,
                  int dimY,
                  int numChannels,
                  float * img_output) {
  size_t input_tensor_size = 3 * 224 * 224;  // simplify ... using known dim values to calculate size
  // size_t input_tensor_size = 3 * 320 * 180;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!
  printf("Loading image\n");

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

  // int width = 320;
  // int height = 180;
  // int width = 224;
  // int height = 224;
  int width = 56;
  int height = 56;
  
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      unsigned char r = *(data++);
      unsigned char g = *(data++);
      unsigned char b = *(data++);

      if (preprocess == "zero") {
        input_tensor_values[(0*width + i)*height + j] = b;
        input_tensor_values[(1*width + i)*height + j] = g;
        input_tensor_values[(2*width + i)*height + j] = r;  
      } 
      else if (preprocess == "unit") {
        input_tensor_values[(0*width + i)*height + j] = b / 255.0;
        input_tensor_values[(1*width + i)*height + j] = g / 255.0;
        input_tensor_values[(2*width + i)*height + j] = r / 255.0;  
      } 
      else if (preprocess == "caffe2") {
        input_tensor_values[(0*width + i)*height + j] = b - 103.939;
        input_tensor_values[(1*width + i)*height + j] = g - 116.779;
        input_tensor_values[(2*width + i)*height + j] = r - 123.68;  
      } 
      else if (preprocess == "caffe") {
        input_tensor_values[(0*width + i)*height + j] = (b - 103.94)*0.017;
        input_tensor_values[(1*width + i)*height + j] = (g - 116.78)*0.017;
        input_tensor_values[(2*width + i)*height + j] = (r - 123.68)*0.017;  
      } else if (preprocess == "mxnet") {
        input_tensor_values[(0*width + i)*height + j] = (b/255.0 - 0.406)/0.225;
        input_tensor_values[(1*width + i)*height + j] = (g/255.0 - 0.456)/0.224;
        input_tensor_values[(2*width + i)*height + j] = (r/255.0 - 0.485)/0.229;  
      } else {
        std::cout << "Unknown preprocess option: " << preprocess << std::endl;
        exit(1);
      }
    }
  }
  // stbi_image_free(orig_data);
  printf("First few image values %f %f %f\n", input_tensor_values[0], input_tensor_values[1], input_tensor_values[2]);

  // initialize input data with values in [0.0, 1.0]
  // for (unsigned int i = 0; i < input_tensor_size; i++)
  //   input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());
  printf("Measuring Cycles...\n");
  auto pre_inference_cycles = read_cycles();

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  auto post_inference_cycles = read_cycles();

  printf("Cycles Elapsed: %llu\n", post_inference_cycles - pre_inference_cycles);

  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  printf("output_tensor\n");
  for(int i = 0; i < output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount(); i++) {
    printf("%d: %f\n", i, floatarr[i]);
  }
  memcpy(img_output, floatarr, sizeof(float)*6);
}

//#include <sys/mman.h> 

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);
  printf("Loaded runner program\n");
  float trailnet_out[6];
  float backup_trailnet_out[6];
  

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
  int intraop = cmd["intra"].as<int>();
  int interop = cmd["inter"].as<int>();
  if (intraop > 1 || interop > 1) {
    session_options.SetExecutionMode(ORT_PARALLEL);
  }
  session_options.SetIntraOpNumThreads(intraop);
  session_options.SetInterOpNumThreads(interop);
  if (cmd.count("trace")) {
    session_options.EnableProfiling(cmd["trace"].as<std::string>().c_str());
  }
  float depth_limit = cmd["wall"].as<float>();
  printf("Using depth: %f\n", depth_limit);
  
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
  const char* backup_model_path = cmd["backupmodel"].as<std::string>().c_str();

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);
  Ort::Session backup_session(env, backup_model_path, session_options);

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

  // int mem_fd;
  // mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
  // ptr = (intptr_t) mmap(NULL, 16, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, mem_fd, 0x2000);
  // printf("Ptr: %x\n", ptr);

    int mem_fd;
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    ptr = (intptr_t) mmap(NULL, 16, PROT_READ 
                                  | PROT_WRITE 
                                  | PROT_EXEC, 
                                    MAP_SHARED, 
                                    mem_fd, 
                                    0x2000);

    printf("Ptr: %x\n", ptr);

  size_t input_tensor_size = 3 * 56 * 56;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!
  printf("Setting target velocity to %f\n", cmd["velocity"].as<float>());
  send_arm();
  send_takeoff();
  // sleep(1);
  printf("Loading image\n");

  int dimX, dimY, numChannels;
  dimX = 56;
  dimY = 56;
  numChannels = 3;

  unsigned char *data = (unsigned char *) malloc(dimX * dimY * numChannels * sizeof(unsigned char));
  int i = 0;
  float depth;
  while(1){
    printf("Processing Image %d...\n", i);


  // score model & input tensor, get back output tensor
  auto pre_load_cycles = read_cycles();
    send_depth_req();
    depth = load_depth();
    loadSimImage(data, dimX, dimY, numChannels);
  auto post_load_cycles = read_cycles();
  printf("Loading Image Cycles Elapsed: %llu\n", post_load_cycles - pre_load_cycles);
  printf("Loaded depth: %f\n", depth);
  // int dimX, dimY, numChannels;
  if (depth < depth_limit) {
    printf("Running small network!\n");
    inferOnImage(data, cmd["preprocess"].as<std::string>(),
                  backup_session, input_node_names, input_node_dims, output_node_names, dimX, dimY, numChannels, trailnet_out);
    if(trailnet_out[0] > trailnet_out[1] && trailnet_out[0] > trailnet_out[2]) {
      trailnet_out[0] = 1.0;
      trailnet_out[1] = 0.0;
      trailnet_out[2] = 0.0;
    } else if (trailnet_out[1] > trailnet_out[2]) {
      trailnet_out[0] = 0.0;
      trailnet_out[1] = 1.0;
      trailnet_out[2] = 0.0;
    } else {
      trailnet_out[0] = 0.0;
      trailnet_out[1] = 0.0;
      trailnet_out[2] = 1.0;
    }
    if(trailnet_out[3] > trailnet_out[4] && trailnet_out[3] > trailnet_out[5]) {
      trailnet_out[3] = 1.0;
      trailnet_out[4] = 0.0;
      trailnet_out[5] = 0.0;
    } else if (trailnet_out[4] > trailnet_out[5]) {
      trailnet_out[3] = 0.0;
      trailnet_out[4] = 1.0;
      trailnet_out[5] = 0.0;
    } else {
      trailnet_out[3] = 0.0;
      trailnet_out[4] = 0.0;
      trailnet_out[5] = 1.0;
    }
  } else {
    printf("Running large network!\n");
    inferOnImage(data, cmd["preprocess"].as<std::string>(),
                  session, input_node_names, input_node_dims, output_node_names, dimX, dimY, numChannels, trailnet_out);
  }

  // unsigned char *data = stbi_load(cmd["image"].as<std::string>().c_str(), &dimX, &dimY, &numChannels, 0);
    // printf("Running first network\n");
    // printf("Running second network\n");

    float yvel   = trailnet_out[3] - trailnet_out[5];
    float yawrate = 2 * (trailnet_out[2] - trailnet_out[0]);
    float zcoord = -1.0;
    float xvel = cmd["velocity"].as<float>();
    // float backup_yvel   = backup_trailnet_out[3] - backup_trailnet_out[5];
    // float backup_yawrate = 2 * (backup_trailnet_out[2] - backup_trailnet_out[0]);
    // float backup_zcoord = -1.0;
    // float backup_xvel = cmd["velocity"].as<float>();
    printf("Target %f, %f, %f, %f...\n", zcoord, xvel, yvel, yawrate);

  auto pre_send_cycles = read_cycles();
    send_target(zcoord, xvel, yvel, yawrate);
  auto post_send_cycles = read_cycles();
  printf("Sending Target Cycles Elapsed: %llu\n", post_send_cycles - pre_send_cycles);
    i++;
  }
  // free(data);
  // send_target(zcoord, xvel, yvel, yawrate);

  return 0;
}

