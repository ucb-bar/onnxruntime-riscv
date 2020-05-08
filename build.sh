#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Setup RISCV environment variables. Ensure that riscv/esp-tools GCC is in your path.
export CXX=riscv64-unknown-linux-gnu-g++
export CC=riscv64-unknown-linux-gnu-gcc
export CXXFLAGS="-march=rv64imafdc -mabi=lp64d"

# Download protoc if we don't have it
if [ ! -d "build/protoc" ]; then
	mkdir -p "build/protoc"
	curl --location "https://github.com/protocolbuffers/protobuf/releases/download/v3.11.2/protoc-3.11.2-linux-x86_64.zip" --output "build/protoc/protoc.zip"
	cd build/protoc
	unzip protoc.zip
fi

cd $DIR

# NOTE: If you're NOT building for the first time adding "--parallel" when invoking this script will parallelize build
# requires python3.6 or higher
python3 $DIR/tools/ci_build/build.py --arm  --build_dir=build "$@"

# On first build, it might fail on linking binary, complaining about missing atomics (this is despite having -latomic).
# Rebuilding fixes this for some reason. This started happening after the version bump subsequent to commit 4db932, as there were some CMake file changes in those commits.

if [ -d "build/Debug" ]; then
	cd build/Debug
	# /scratch/pranavprakash/chipyard/esp-tools-install/bin/riscv64-unknown-linux-gnu-g++  -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-parentheses -g -Wno-nonnull-compare -Wno-deprecated-copy  -latomic -static CMakeFiles/onnx_test_runner.dir/scratch/pranavprakash/onnxruntime/onnxruntime/onnxruntime/test/onnx/main.cc.o  -o onnx_test_runner  libonnx_test_runner_common.a libonnxruntime_test_utils.a libonnxruntime_session.a libonnxruntime_providers_systolic.a libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_util.a libonnxruntime_framework.a libonnxruntime_util.a libonnxruntime_graph.a libonnxruntime_common.a libonnxruntime_mlas.a libonnx_test_data_proto.a external/re2/libre2.a onnx/libonnx.a onnx/libonnx_proto.a external/protobuf/cmake/libprotobuf-lited.a external/re2/libre2.a external/nsync/libnsync_cpp.a -ldl -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive
else
	# If we're release build don't bother linking the test runner binaries
	cd build/Release
fi

# Statically link lpthread, latomic, lrt into the test-runner binary so we can run using riscv userspace qemu
#riscv64-unknown-linux-gnu-g++  -march=rv64imafdc -mabi=lp64 -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-parentheses -g -Wno-nonnull-compare  -latomic -static CMakeFiles/onnx_test_runner.dir/scratch/pranavprakash/onnxruntime/onnxruntime/onnxruntime/test/onnx/main.cc.o  -o onnx_test_runner  libonnx_test_runner_common.a libonnxruntime_test_utils.a libonnxruntime_session.a libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_util.a libonnxruntime_framework.a libonnxruntime_util.a libonnxruntime_graph.a libonnxruntime_common.a libonnxruntime_providers_systolic.a libonnxruntime_mlas.a libonnx_test_data_proto.a external/re2/libre2.a onnx/libonnx.a onnx/libonnx_proto.a external/protobuf/cmake/libprotobuf-lited.a external/re2/libre2.a -ldl external/protobuf/cmake/libprotobufd.a -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive

#Old version for commit 4db932
#riscv64-unknown-linux-gnu-g++ -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-parentheses -g -Wno-nonnull-compare  -latomic CMakeFiles/onnx_test_runner.dir/scratch/pranavprakash/onnxruntime/onnxruntime/onnxruntime/test/onnx/main.cc.obj  -o onnx_test_runner  libonnx_test_runner_common.a libonnxruntime_test_utils.a libonnxruntime_session.a libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_util.a libonnxruntime_framework.a libonnxruntime_util.a libonnxruntime_graph.a libonnxruntime_common.a libonnxruntime_mlas.a libonnx_test_data_proto.a external/re2/libre2.a onnx/libonnx.a onnx/libonnx_proto.a external/protobuf/cmake/libprotobuf-lited.a external/re2/libre2.a -ldl external/protobuf/cmake/libprotobufd.a -static -Wl,--whole-archive -lpthread -latomic -Wl,--no-whole-archive


# For MNN
#cmake .. -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_VERSION=1 -DCMAKE_SYSTEM_PROCESSOR=riscv -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_EXE_LINKER_FLAGS=" -static -Wl,--whole-archive -lpthread -latomic -Wl,--no-whole-archive"

/scratch/pranavprakash/chipyard/esp-tools-install/bin/riscv64-unknown-linux-gnu-g++  -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-deprecated-copy -Wno-parentheses -O3 -DNDEBUG -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION  -latomic -static CMakeFiles/onnxruntime_training_mnist.dir/scratch/pranavprakash/onnxruntime/onnxruntime/orttraining/orttraining/models/mnist/main.cc.o CMakeFiles/onnxruntime_training_mnist.dir/scratch/pranavprakash/onnxruntime/onnxruntime/orttraining/orttraining/models/mnist/mnist_data_provider.cc.o  -o onnxruntime_training_mnist  libonnxruntime_training_runner.a libonnxruntime_training.a libonnxruntime_session.a libonnxruntime_optimizer.a libonnxruntime_providers.a libonnxruntime_util.a libonnxruntime_framework.a libonnxruntime_util.a libonnxruntime_graph.a libonnxruntime_common.a libonnxruntime_mlas.a onnx/libonnx.a onnx/libonnx_proto.a external/protobuf/cmake/libprotobuf-lite.a external/re2/libre2.a tensorboard/libtensorboard.a external/nsync/libnsync_cpp.a -ldl -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive