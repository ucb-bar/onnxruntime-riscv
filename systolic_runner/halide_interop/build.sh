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
    make "bin/runtime.generator"
    cd bin
    ./runtime.generator -r halide_runtime -o . target="riscv-64-linux"
    cd ..
    cd "apps/onnx"
    make model_test
fi