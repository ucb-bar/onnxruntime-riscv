#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OS=$(uname -s)

# Setup RISCV environment variables. Ensure that riscv/esp-tools GCC is in your path.
# By default, we assume we are building for systolic only, so riscv-tools suffices with standard target
export CXX=riscv64-unknown-linux-gnu-g++
export CC=riscv64-unknown-linux-gnu-gcc
export CXXFLAGS="-march=rv64imafdc -mabi=lp64d"

BUILD_TYPE="Debug"
for var in "$@"
do
	if [ $var = "--config=Release" ]; then
		BUILD_TYPE="Release"
	fi
	if [ $var = "--use_hwacha" ]; then
		echo "Building with hwacha support"
		# Note that CFLAGS needs to be set for assembler to pickup
		export CXXFLAGS="-march=rv64gcxhwacha -mabi=lp64d"
		export CFLAGS="-march=rv64gcxhwacha -mabi=lp64d"
	fi
done


echo "Performing ${BUILD_TYPE} build"

# Download protoc if we don't have it
if [ ! -d "build/protoc" ]; then
	mkdir -p "build/protoc"
	curl --location "https://github.com/protocolbuffers/protobuf/releases/download/v3.16.0/protoc-3.16.0-linux-x86_64.zip" --output "build/protoc/protoc.zip"
	cd build/protoc
	unzip protoc.zip
fi

cd $DIR

# NOTE: If you're NOT building for the first time adding "--parallel" when invoking this script will parallelize build
# requires python3.6 or higher
python3 $DIR/tools/ci_build/build.py --riscv --skip_submodule_sync --update --build --build_dir=build "$@"


# Note that if you ever want to use the onnx_test_runner then you'll probably have to uncomment the below,
# Since we need to relink with pthreads as --whole-archive. See the comment for Mlas which applies here.
# But we never really use that, so for now just ignore.
if [ ${BUILD_TYPE} = "Release" ]; then
	cd build/Release
	# Statically link lpthread, latomic, lrt into the test-runner binary so we can run using riscv userspace qemu
	# /scratch/pranavprakash/chipyard/esp-tools-install/bin/riscv64-unknown-linux-gnu-g++  -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-parentheses -g -Wno-nonnull-compare -Wno-deprecated-copy  -latomic -static CMakeFiles/onnx_test_runner.dir/scratch/pranavprakash/onnxruntime/onnxruntime/onnxruntime/test/onnx/main.cc.o  -o onnx_test_runner  libonnx_test_runner_common.a libonnxruntime_test_utils.a libonnxruntime_session.a libonnxruntime_providers_systolic.a libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_util.a libonnxruntime_framework.a libonnxruntime_util.a libonnxruntime_graph.a libonnxruntime_common.a libonnxruntime_mlas.a libonnx_test_data_proto.a external/re2/libre2.a onnx/libonnx.a onnx/libonnx_proto.a external/protobuf/cmake/libprotobuf-lited.a external/re2/libre2.a external/nsync/libnsync_cpp.a -ldl -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive
else
	# If we're release build don't bother linking the test runner binaries
	cd build/Debug
fi

## !!!!IMPORTANT: MLAS TEST RUNNER WILL FAIL ON MULTITHREAD UNLESS YOU UNCOMMENT BELOW!!!

# We statically link the binary but everyone in the linux world (except musl libc I guess) seems to be bent
# on making this as hard as possible. In particular we need to take care with pthreads and ensure that
# weak-links are replaced by using --whole-archive. See
# https://stackoverflow.com/questions/35116327/when-g-static-link-pthread-cause-segmentation-fault-why
# and other related threads for more info.

# Statically link lpthread, latomic, lrt into the mlas unit test
# ${CXX} ${CXXFLAGS} -Wno-error=attributes -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-deprecated-copy -Wno-parentheses -O2 -DNDEBUG -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION -static CMakeFiles/onnxruntime_mlas_test.dir/home/centos/onnxruntime-riscv/onnxruntime/test/mlas/unittest.cpp.o  -o onnxruntime_mlas_test libonnxruntime_mlas.a libonnxruntime_common.a external/nsync/libnsync_cpp.a -ldl -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive


## Outdated, ignore all below

#Old version for commit 4db932
#riscv64-unknown-linux-gnu-g++ -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-parentheses -g -Wno-nonnull-compare  -latomic CMakeFiles/onnx_test_runner.dir/scratch/pranavprakash/onnxruntime/onnxruntime/onnxruntime/test/onnx/main.cc.obj  -o onnx_test_runner  libonnx_test_runner_common.a libonnxruntime_test_utils.a libonnxruntime_session.a libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_util.a libonnxruntime_framework.a libonnxruntime_util.a libonnxruntime_graph.a libonnxruntime_common.a libonnxruntime_mlas.a libonnx_test_data_proto.a external/re2/libre2.a onnx/libonnx.a onnx/libonnx_proto.a external/protobuf/cmake/libprotobuf-lited.a external/re2/libre2.a -ldl external/protobuf/cmake/libprotobufd.a -static -Wl,--whole-archive -lpthread -latomic -Wl,--no-whole-archive

# For MNN
#cmake .. -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=riscv -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_EXE_LINKER_FLAGS=" -static -Wl,--whole-archive -lpthread -latomic -Wl,--no-whole-archive"
