#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

release_path="Debug"
if [ $1 = "--config=Release" ]; then
	release_path="Release"
fi


# I clearly have no idea how to actually write bash scripts.
echo $1
root_path="`cd "../../../";pwd`"
build_path=${root_path}/build/${release_path}

extra_libs=""
extra_defs=""
if [ -f "${root_path}/systolic_runner/halide_interop/model_converter/generated/libcustom.a" ]; then
    echo "Found custom lib. Building with support for loading it."
    extra_libs="${root_path}/systolic_runner/halide_interop/model_converter/generated/libcustom.a \
                ${root_path}/systolic_runner/halide_interop/Halide/bin/halide_runtime.a"
    extra_defs="-DUSE_CUSTOM_OP_LIBRARY"
fi

riscv64-unknown-linux-gnu-g++ -O3 -I ${root_path}/include/onnxruntime/core/session -I  ${root_path}/include/onnxruntime/core/providers -march=rv64imafdc -mabi=lp64d -Wno-error=attributes \
 -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS ${extra_defs} -Wall -Wextra -ffunction-sections -fdata-sections -Wno-parentheses -g -Wno-nonnull-compare \
  -latomic -static runner.cpp  -o ort_test  ${build_path}/libonnx_test_runner_common.a ${build_path}/libonnxruntime_test_utils.a \
   ${build_path}/libonnxruntime_session.a ${build_path}/libonnxruntime_optimizer.a ${build_path}/libonnxruntime_providers.a \
    ${build_path}/libonnxruntime_util.a ${build_path}/libonnxruntime_framework.a ${build_path}/libonnxruntime_util.a \
     ${build_path}/libonnxruntime_graph.a ${build_path}/libonnxruntime_providers_systolic.a ${build_path}/libonnxruntime_common.a \
     ${build_path}/libonnxruntime_mlas.a ${extra_libs} \
      ${build_path}/libonnx_test_data_proto.a ${build_path}/external/re2/libre2.a ${build_path}/external/nsync/libnsync_cpp.a ${build_path}/onnx/libonnx.a \
       ${build_path}/onnx/libonnx_proto.a ${build_path}/external/protobuf/cmake/libprotobuf-lite*.a ${build_path}/external/re2/libre2.a \
        -ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive