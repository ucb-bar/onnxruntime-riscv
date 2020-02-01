#!/bin/bash
riscv64-unknown-linux-gnu-g++ -I ../include/onnxruntime/core/session -I ../include/onnxruntime/core/providers -march=rv64imafdc -mabi=lp64d -Wno-error=attributes \
 -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wall -Wextra -ffunction-sections -fdata-sections -Wno-parentheses -g -Wno-nonnull-compare \
  -latomic -static runner.cpp  -o ort_test  ../build/Debug/libonnx_test_runner_common.a ../build/Debug/libonnxruntime_test_utils.a \
   ../build/Debug/libonnxruntime_session.a ../build/Debug/libonnxruntime_optimizer.a ../build/Debug/libonnxruntime_providers.a \
    ../build/Debug/libonnxruntime_util.a ../build/Debug/libonnxruntime_framework.a ../build/Debug/libonnxruntime_util.a \
     ../build/Debug/libonnxruntime_graph.a ../build/Debug/libonnxruntime_providers_systolic.a ../build/Debug/libonnxruntime_common.a \
     ../build/Debug/libonnxruntime_mlas.a \
      ../build/Debug/libonnx_test_data_proto.a ../build/Debug/external/re2/libre2.a ../build/Debug/onnx/libonnx.a \
       ../build/Debug/onnx/libonnx_proto.a ../build/Debug/external/protobuf/cmake/libprotobuf-lited.a ../build/Debug/external/re2/libre2.a \
        -ldl -static -Wl,--whole-archive -lpthread -latomic -lrt -Wl,--no-whole-archive

