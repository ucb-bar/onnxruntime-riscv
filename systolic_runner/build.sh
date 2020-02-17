#!/bin/bash
build_path="Debug"
if [ $1 = "--config=Release" ]; then
	build_path="Release"
fi

echo $1
echo $build_path

riscv64-unknown-linux-gnu-g++ -O3 -I ../include/onnxruntime/core/session -I ../include/onnxruntime/core/providers -march=rv64imafdc -mabi=lp64d -Wno-error=attributes \
 -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Wno-parentheses -g -Wno-nonnull-compare \
  -latomic -static src/runner.cpp  -o ort_test  ../build/${build_path}/libonnx_test_runner_common.a ../build/${build_path}/libonnxruntime_test_utils.a \
   ../build/${build_path}/libonnxruntime_session.a ../build/${build_path}/libonnxruntime_optimizer.a ../build/${build_path}/libonnxruntime_providers.a \
    ../build/${build_path}/libonnxruntime_util.a ../build/${build_path}/libonnxruntime_framework.a ../build/${build_path}/libonnxruntime_util.a \
     ../build/${build_path}/libonnxruntime_graph.a ../build/${build_path}/libonnxruntime_providers_systolic.a ../build/${build_path}/libonnxruntime_common.a \
     ../build/${build_path}/libonnxruntime_mlas.a \
      ../build/${build_path}/libonnx_test_data_proto.a ../build/${build_path}/external/re2/libre2.a ../build/${build_path}/external/nsync/libnsync_cpp.a ../build/${build_path}/onnx/libonnx.a \
       ../build/${build_path}/onnx/libonnx_proto.a ../build/${build_path}/external/protobuf/cmake/libprotobuf-lite*.a ../build/${build_path}/external/re2/libre2.a \
        -ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive

