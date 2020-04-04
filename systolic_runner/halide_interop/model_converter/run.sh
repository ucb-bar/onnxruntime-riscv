#!/bin/bash

PYTHONPATH=../Halide/apps/onnx/bin/host python3 converter.py "$@"

if [ -d "generated" ]; then
cd "generated"

riscv64-unknown-linux-gnu-g++ -fPIC -DDISABLE_CONTRIB_OPS -DEIGEN_MPL2_ONLY -DPLATFORM_POSIX -DUSE_EIGEN_FOR_BLAS -DUSE_SYSTOLIC=1 \
 -Dcustom_op_library_EXPORTS -I../../../..//include/onnxruntime -I../../Halide/src/runtime -I../../../..//include/onnxruntime/core/session \
    -I../../../..//include -static -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS \
     -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-deprecated-copy -Wno-parentheses -Wno-missing-field-initializers -O3 -DNDEBUG \
    -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION -fPIC -std=gnu++14 -c custom_op_library.cc 

riscv64-unknown-linux-gnu-g++ -fPIC -I../../Halide/src/runtime  \
    -I../../../..//include -static -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS \
     -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-deprecated-copy -Wno-unused-variable -Wno-parentheses -Wno-unknown-pragmas -Wno-missing-field-initializers -O3 -DNDEBUG \
    -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION -fPIC -std=gnu++14 -c *.cpp
    
riscv64-unknown-linux-gnu-ar rcs libcustom.a *.o
fi

