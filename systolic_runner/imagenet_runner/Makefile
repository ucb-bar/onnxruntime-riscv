CXX = riscv64-unknown-linux-gnu-g++
CFLAGS = -march=rv64imafdc -mabi=lp64d -Wno-error=attributes \
   	-Wall -Wextra -ffunction-sections -fdata-sections -Wno-parentheses -g -Wno-nonnull-compare \
  	-latomic -static
CDEFS = -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS ${extra_defs}

stb_image.o:
	$(CXX) -c -o src/stb_image.o src/stb_image.cpp $(CFLAGS) $(CDEFS)

ort_test: stb_image.o
	$(CXX) -o ort_test src/runner.cpp src/stb_image.o $(CDEFS) \
				-I ${root_path}/include/onnxruntime/core/session \
				-I ${root_path}/include/onnxruntime/core/providers \
				-I ${root_path}/cmake/external/cxxopts/include \
				-I ${root_path}/orttraining \
				-I ${root_path}/include/onnxruntime \
				-I ${root_path}/cmake/external/optional-lite/include \
				-I ${root_path}/core \
				-I ${build_path} \
				-I ${build_path}/external/onnx \
				-I ${root_path}/cmake/external/onnx \
				-I ${root_path}/cmake/external/nsync/public \
				-I ${root_path}/cmake/external/SafeInt \
				-I ${root_path}/onnxruntime \
				-I ${root_path}/cmake/external/protobuf/src \
				$(CFLAGS) ${build_path}/libonnx_test_runner_common.a ${build_path}/libonnxruntime_test_utils.a \
   ${training_libs} \
   ${build_path}/libonnxruntime_session.a ${build_path}/libonnxruntime_optimizer.a ${build_path}/libonnxruntime_providers.a \
    ${build_path}/libonnxruntime_util.a ${build_path}/libonnxruntime_framework.a ${build_path}/libonnxruntime_util.a \
     ${build_path}/libonnxruntime_graph.a ${build_path}/libonnxruntime_providers_systolic.a ${extra_providers} ${build_path}/libonnxruntime_common.a \
     ${build_path}/libonnxruntime_mlas.a ${build_path}/libonnxruntime_flatbuffers.a ${extra_libs} \
      ${build_path}/libonnx_test_data_proto.a ${build_path}/external/re2/libre2.a ${build_path}/external/nsync/libnsync_cpp.a ${build_path}/external/onnx/libonnx.a \
       ${build_path}/external/onnx/libonnx_proto.a ${build_path}/external/protobuf/cmake/libprotobuf-lite*.a ${build_path}/external/re2/libre2.a \
       ${build_path}/external/flatbuffers/libflatbuffers.a  -ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive