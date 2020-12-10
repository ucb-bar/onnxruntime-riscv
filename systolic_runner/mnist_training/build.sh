#!/bin/bash
release_path="Debug"

# First determine what path to link against
for var in "$@"
do
    if [ $var = "--config=Release" ]; then
        release_path="Release"
    fi
done
# I clearly have no idea how to actually write bash scripts. At all.
echo "Building against ${release_path}"
root_path=../../
build_path=${root_path}/build/${release_path}

extra_libs=""
extra_defs=""
training_enabled=""

# Check for hwacha support
for var in "$@"
do
	if [ $var = "--use_hwacha" ]; then
		echo "Building with hwacha support"
        extra_defs="${extra_defs} -DUSE_HWACHA"
        extra_libs="${extra_libs} ${build_path}/libonnxruntime_providers_hwacha.a"
	fi
    if [ $var = "--for_firesim" ]; then 
        echo "Building with mlockall for running on Firesim"
        extra_defs="${extra_defs} -DFOR_FIRESIM"
    fi
    if [ $var = "--enable_training" ]; then
        extra_libs="${extra_libs} ${build_path}/tensorboard/libtensorboard.a"
        training_enabled="YES"
    fi
done

if [[ ! $training_enabled ]]; then
    echo "Must build with --enable_training "
    exit 1
fi

# Check for halide interop generated custom operators
if [ -f "${root_path}/systolic_runner/halide_interop/model_converter/generated/libcustom.a" ]; then
    echo "Found custom lib. Building with support for loading it."
    extra_libs="${extra_libs} ${root_path}/systolic_runner/halide_interop/model_converter/generated/libcustom.a \
                ${root_path}/systolic_runner/halide_interop/Halide/bin/halide_runtime.a"
    extra_defs="${extra_defs} -DUSE_CUSTOM_OP_LIBRARY"
fi

riscv64-unknown-linux-gnu-g++ -O3 -I ${root_path}/include/onnxruntime/core/session \
-DEIGEN_MPL2_ONLY -DEIGEN_USE_THREADS -DENABLE_ORT_FORMAT_LOAD \
-DENABLE_TRAINING -DNSYNC_ATOMIC_CPP11 -DONNX_ML=1 -DONNX_NAMESPACE=onnx \
-DONNX_USE_LITE_PROTO=1 -DPLATFORM_POSIX -DSYSTOLIC_INT8=1 -DUSE_EIGEN_FOR_BLAS \
-DUSE_SYSTOLIC=1 -D__ONNX_NO_DOC_STRINGS  \
-I  ${root_path}/include/onnxruntime/core/providers \
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
-march=rv64imafdc -mabi=lp64d -Wno-error=attributes \
 -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS ${extra_defs} -Wall -Wextra -ffunction-sections -fdata-sections -Wno-parentheses -g -Wno-nonnull-compare \
  -latomic -static main.cc mnist_data_provider.cc -o mnist_train  ${build_path}/libonnx_test_runner_common.a ${build_path}/libonnxruntime_test_utils.a \
   ${build_path}/libonnxruntime_training_runner.a ${build_path}/libonnxruntime_training.a \
   ${build_path}/libonnxruntime_session.a ${build_path}/libonnxruntime_optimizer.a ${build_path}/libonnxruntime_providers.a \
    ${build_path}/libonnxruntime_util.a ${build_path}/libonnxruntime_framework.a ${build_path}/libonnxruntime_util.a \
     ${build_path}/libonnxruntime_graph.a ${build_path}/libonnxruntime_providers_systolic.a ${build_path}/libonnxruntime_common.a \
     ${build_path}/libonnxruntime_mlas.a ${build_path}/libonnxruntime_flatbuffers.a ${extra_libs} \
      ${build_path}/libonnx_test_data_proto.a ${build_path}/external/re2/libre2.a ${build_path}/external/nsync/libnsync_cpp.a ${build_path}/external/onnx/libonnx.a \
       ${build_path}/external/onnx/libonnx_proto.a ${build_path}/external/protobuf/cmake/libprotobuf-lite*.a ${build_path}/external/re2/libre2.a \
       ${build_path}/external/flatbuffers/libflatbuffers.a  -ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive