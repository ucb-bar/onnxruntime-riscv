
def custom_format(string, brackets, *args, **kwargs):
    if len(brackets) != 2:
        raise ValueError('Expected two brackets. Got {}.'.format(len(brackets)))
    padded = string.replace('{', '{{').replace('}', '}}')
    substituted = padded.replace(brackets[0], '{').replace(brackets[1], '}')
    formatted = substituted.format(*args, **kwargs)
    return formatted

header = """
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);


#ifdef __cplusplus
}
#endif

#define EXCLUDE_REFERENCE_TO_ORT_DLL
#include "onnxruntime_cxx_api.h"
#undef EXCLUDE_REFERENCE_TO_ORT_DLL

#include <vector>
#include <cmath>
#include "HalideBuffer.h"

static const char* c_OpDomain = "test.customop";

std::vector<halide_dimension_t> getHalideDimsForVector(const std::vector<int64_t> &shape) {
  // Hacky workaround for fact that halide indexes things differently
  std::vector<halide_dimension_t> ret(shape.size());
  int32_t cum = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    ret[i] = {0, (int32_t) shape[i], cum};
    cum *= shape[i];
  }
  return ret;
}

std::vector<halide_dimension_t> getTensorDims(Ort::CustomOpApi ort, const OrtValue* value) {
  OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
  std::vector<int64_t> shape = ort.GetTensorShape(info);
  ort.ReleaseTensorTypeAndShapeInfo(info);
  return getHalideDimsForVector(shape);
}

template<typename T>
Halide::Runtime::Buffer<const T> getInBufferForOrtValue(Ort::CustomOpApi ort, const OrtValue* value) {
  return Halide::Runtime::Buffer<const T>(ort.GetTensorData<T>(value), getTensorDims(ort, value));
}

template<typename T>
Halide::Runtime::Buffer<T> getOutBufferForOrtValue(Ort::CustomOpApi ort, OrtValue* value, const std::vector<int64_t> &dims) {
  return Halide::Runtime::Buffer<T>(ort.GetTensorMutableData<T>(value), getHalideDimsForVector(dims));
}
"""

kernel_def = """
struct Kernel@kernel_name$ {
  Kernel@kernel_name$(OrtApi api)
     :api_(api),
     ort_(api_)
  {
  }

  void Compute(OrtKernelContext* context) {
    printf("Called into custom op library for @kernel_name$\\n");

    {// Setup inputs
    @setup_input$
    @setup_output$

    @call_function$
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};
"""

op_def = """
struct CustomOp@kernel_name$ : Ort::CustomOpBase<CustomOp@kernel_name$, Kernel@kernel_name$> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new Kernel@kernel_name$(api);
  };

  const char* GetName() const { return "CustomOp@kernel_name$"; };

  size_t GetInputTypeCount() const { return @num_inputs$; };
  ONNXTensorElementDataType GetInputType(size_t index) const { 
      @input_type$ 
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; // default
    };

  size_t GetOutputTypeCount() const { return @num_outputs$; };
  ONNXTensorElementDataType GetOutputType(size_t index) const { 
      @output_type$
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; // default
    };

} c_CustomOp@kernel_name$;
"""

register_single = """
  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOp@kernel_name$)) {
    return status;
  }
"""

register = """
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  @register_kernels$

  return ortApi->AddCustomOpDomain(options, domain);
}
"""