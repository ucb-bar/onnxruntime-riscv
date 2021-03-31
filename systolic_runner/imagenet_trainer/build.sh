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
extra_providers=""
# Only set if we build with --enable_training
training_libs=""

for var in "$@"
do
	if [ $var = "--use_hwacha" ]; then
		echo "Building with hwacha support"
        extra_defs="-DUSE_HWACHA ${extra_defs}"
        extra_providers="${build_path}/libonnxruntime_providers_hwacha.a ${extra_providers}"
	fi
    if [ $var = "--for_firesim" ]; then 
        echo "Building with mlockall for running on Firesim"
        extra_defs="-DFOR_FIRESIM ${extra_defs}"
    fi
    if [ $var = "--enable_training" ]; then
        extra_libs="${build_path}/tensorboard/libtensorboard.a ${extra_libs}"
        training_libs="${build_path}/libonnxruntime_training_runner.a ${build_path}/libonnxruntime_training.a"
    fi

done

# Check for halide interop generated custom operators
if [ -f "${root_path}/systolic_runner/halide_interop/model_converter/generated/libcustom.a" ]; then
    echo "Found custom lib. Building with support for loading it."
    extra_libs="${root_path}/systolic_runner/halide_interop/model_converter/generated/libcustom.a \
                ${root_path}/systolic_runner/halide_interop/Halide/bin/halide_runtime.a ${extra_libs}"
    extra_defs="-DUSE_CUSTOM_OP_LIBRARY ${extra_defs}"
fi

# Clean the old binary since we never really invoke this script unless we want to force a build
rm -f resnet_train
# 16 cores is surely overkill for 2 jobs
make -s -j16 resnet_train root_path="${root_path}" build_path="${build_path}" extra_libs="${extra_libs}" \
                       extra_defs="${extra_defs}" training_libs="${training_libs}" extra_providers="${extra_providers}"
echo "Please ignore any dlopen warning above. Glibc hates being statically linked."