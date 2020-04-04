#include "custom_op_library.h"

#define EXCLUDE_REFERENCE_TO_ORT_DLL
#include "onnxruntime_cxx_api.h"
#undef EXCLUDE_REFERENCE_TO_ORT_DLL

#include <vector>
#include <cmath>
#include "HalideBuffer.h"

static const char* c_OpDomain = "test.customop";

std::vector<int64_t> getTensorDims(Ort::CustomOpApi ort, const OrtValue* value) {
  OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
  std::vector<int64_t> ret = ort.GetTensorShape(info);
  ort.ReleaseTensorTypeAndShapeInfo(info);
  return ret;
}

template typename<T>
Halide::Runtime::Buffer<T> getBufferForOrtValue(Ort::CustomOpApi ort, const OrtValue* value) {
  return Halide::Runtime::Buffer<T>(ort.GetTensorData<T>(value), getTensorDims(ort, value))
}

template typename<T>
Halide::Runtime::Buffer<T> getBufferForOrtValue(Ort::CustomOpApi ort, const OrtValue* value, const std::vector<int64_t> &dims) {
  return Halide::Runtime::Buffer<T>(ort.GetTensorData<T>(value), dims)
}

struct KernelOne {
  KernelOne(OrtApi api)
     :api_(api),
     ort_(api_)
  {
  }

  void Compute(OrtKernelContext* context) {
    printf("Called into custom op library\n");

    {// Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
    const float* X = ort_.GetTensorData<float>(input_X);
    const float* Y = ort_.GetTensorData<float>(input_Y);
    
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dims.data(), dims.size());
    float* out = ort_.GetTensorMutableData<float>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};


struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelOne(api);
  };

  const char* GetName() const { return "CustomOpOne"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

} c_CustomOpOne;


OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpOne)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}