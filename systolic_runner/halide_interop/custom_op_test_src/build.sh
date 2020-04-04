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

# Build the custom op lib
riscv64-unknown-linux-gnu-g++ -fPIC -DDISABLE_CONTRIB_OPS -DEIGEN_MPL2_ONLY -DPLATFORM_POSIX -DUSE_EIGEN_FOR_BLAS -DUSE_SYSTOLIC=1 \
 -Dcustom_op_library_EXPORTS -I${root_path}/include/onnxruntime -I${root_path}/include/onnxruntime/core/session \
    -I${root_path}/include -static -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS \
     -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-deprecated-copy -Wno-parentheses -O3 -DNDEBUG \
    -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION -fPIC -std=gnu++14 -o custom_op_library.cc.o -c custom_op_library.cc 

riscv64-unknown-linux-gnu-g++ -O3 -I ${root_path}/include/onnxruntime/core/session -I  ${root_path}/include/onnxruntime/core/providers -march=rv64imafdc -mabi=lp64d -Wno-error=attributes \
 -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Wno-parentheses -g -Wno-nonnull-compare \
  -latomic -static runner.cpp custom_op_library.cc.o  -o ort_test  ${build_path}/libonnx_test_runner_common.a ${build_path}/libonnxruntime_test_utils.a \
   ${build_path}/libonnxruntime_session.a ${build_path}/libonnxruntime_optimizer.a ${build_path}/libonnxruntime_providers.a \
    ${build_path}/libonnxruntime_util.a ${build_path}/libonnxruntime_framework.a ${build_path}/libonnxruntime_util.a \
     ${build_path}/libonnxruntime_graph.a ${build_path}/libonnxruntime_providers_systolic.a ${build_path}/libonnxruntime_common.a \
     ${build_path}/libonnxruntime_mlas.a \
      ${build_path}/libonnx_test_data_proto.a ${build_path}/external/re2/libre2.a ${build_path}/external/nsync/libnsync_cpp.a ${build_path}/onnx/libonnx.a \
       ${build_path}/onnx/libonnx_proto.a ${build_path}/external/protobuf/cmake/libprotobuf-lite*.a ${build_path}/external/re2/libre2.a \
        -ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive

# riscv64-unknown-linux-gnu-g++ -O3 -I ${root_path}/include/onnxruntime/core/session -I  ${root_path}/include/onnxruntime/core/providers -march=rv64imafdc -mabi=lp64d -Wno-error=attributes \
#  -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Wno-parentheses -g -Wno-nonnull-compare \
#   -latomic -static imagenet_halide/runner.cpp model_converter/generated/libcustom.a  -o ort_test  ${build_path}/libonnx_test_runner_common.a ${build_path}/libonnxruntime_test_utils.a \
#    ${build_path}/libonnxruntime_session.a ${build_path}/libonnxruntime_optimizer.a ${build_path}/libonnxruntime_providers.a \
#     ${build_path}/libonnxruntime_util.a ${build_path}/libonnxruntime_framework.a ${build_path}/libonnxruntime_util.a \
#      ${build_path}/libonnxruntime_graph.a ${build_path}/libonnxruntime_providers_systolic.a ${build_path}/libonnxruntime_common.a \
#      ${build_path}/libonnxruntime_mlas.a \
#       ${build_path}/libonnx_test_data_proto.a ${build_path}/external/re2/libre2.a ${build_path}/external/nsync/libnsync_cpp.a ${build_path}/onnx/libonnx.a \
#        ${build_path}/onnx/libonnx_proto.a ${build_path}/external/protobuf/cmake/libprotobuf-lite*.a ${build_path}/external/re2/libre2.a \
#         -ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive

rm custom_op_library.cc.o # Intermediate we don't need


# riscv64-unknown-linux-gnu-g++ -fPIC -DDISABLE_CONTRIB_OPS -DEIGEN_MPL2_ONLY -DPLATFORM_POSIX -DUSE_EIGEN_FOR_BLAS -DUSE_SYSTOLIC=1 \
#  -Dcustom_op_library_EXPORTS -I../../../..//include/onnxruntime -I../../Halide/src/runtime -I../../../..//include/onnxruntime/core/session \
#     -I../../../..//include -static -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS \
#      -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-deprecated-copy -Wno-parentheses -Wno-missing-field-initializers -O3 -DNDEBUG \
#     -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION -fPIC -std=gnu++14 -c custom_op_library.cc 

#     riscv64-unknown-linux-gnu-g++ -fPIC -I../../Halide/src/runtime  \
#     -I../../../..//include -static -march=rv64imafdc -mabi=lp64d -Wno-error=attributes -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS \
#      -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-deprecated-copy -Wno-unused-variable -Wno-parentheses -Wno-unknown-pragmas -Wno-missing-field-initializers -O3 -DNDEBUG \
#     -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION -fPIC -std=gnu++14 -c *.cpp