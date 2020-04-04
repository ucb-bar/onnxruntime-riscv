#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

release_path="Debug"
if [ $1 = "--config=Release" ]; then
	release_path="Release"
fi


# I clearly have no idea how to actually write bash scripts.
echo $1
root_path="`cd "../../";pwd`"
build_path=${root_path}/build/${release_path}

# Download fresh clang if we don't have it
if [ ! -d "${root_path}/build/llvm" ]; then
    echo "Downloading clang..."
	mkdir -p "${root_path}/build/llvm"
	curl --location "https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz" --output "${root_path}/build/llvm/llvm.tar.xz"
	cd ${root_path}/build/llvm
    tar xvf llvm.tar.xz
    rm llvm.tar.xz
fi

cd $DIR

# Download fresh clang if we don't have it
if [ ! -d "Halide/bin" ]; then
    cd "Halide"
    echo "Building Halide..."
    export LLVM_CONFIG="${root_path}/build/llvm/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/llvm-config"
    make -j16
fi

# Build the custom op lib
# riscv64-unknown-linux-gnu-g++ -fPIC -DDISABLE_CONTRIB_OPS -DEIGEN_MPL2_ONLY -DPLATFORM_POSIX -DUSE_EIGEN_FOR_BLAS -DUSE_SYSTOLIC=1 \
#  -Dcustom_op_library_EXPORTS -I${root_path}/include/onnxruntime -I${root_path}/include/onnxruntime/core/session \
#     -I${root_path}/include -static -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS \
#      -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-deprecated-copy -Wno-parentheses -O3 -DNDEBUG \
#     -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION -fPIC -std=gnu++14 -o custom_op_library.cc.o -c src/custom_op_library.cc 

# riscv64-unknown-linux-gnu-g++ -O3 -I ${root_path}/include/onnxruntime/core/session -I  ${root_path}/include/onnxruntime/core/providers -march=rv64imafdc -mabi=lp64d -Wno-error=attributes \
#  -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Wno-parentheses -g -Wno-nonnull-compare \
#   -latomic -static src/runner.cpp custom_op_library.cc.o  -o ort_test  ${build_path}/libonnx_test_runner_common.a ${build_path}/libonnxruntime_test_utils.a \
#    ${build_path}/libonnxruntime_session.a ${build_path}/libonnxruntime_optimizer.a ${build_path}/libonnxruntime_providers.a \
#     ${build_path}/libonnxruntime_util.a ${build_path}/libonnxruntime_framework.a ${build_path}/libonnxruntime_util.a \
#      ${build_path}/libonnxruntime_graph.a ${build_path}/libonnxruntime_providers_systolic.a ${build_path}/libonnxruntime_common.a \
#      ${build_path}/libonnxruntime_mlas.a \
#       ${build_path}/libonnx_test_data_proto.a ${build_path}/external/re2/libre2.a ${build_path}/external/nsync/libnsync_cpp.a ${build_path}/onnx/libonnx.a \
#        ${build_path}/onnx/libonnx_proto.a ${build_path}/external/protobuf/cmake/libprotobuf-lite*.a ${build_path}/external/re2/libre2.a \
#         -ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive

rm custom_op_library.cc.o # Intermediate we don't need
