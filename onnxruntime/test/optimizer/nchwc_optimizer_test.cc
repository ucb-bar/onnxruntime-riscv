// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/compare_ortvalue.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// InferenceSession wrapper in order to gain access to the loaded graph.
class NchwcInferenceSession : public InferenceSession {
 public:
  explicit NchwcInferenceSession(const SessionOptions& session_options,
                                 logging::LoggingManager* logging_manager) : InferenceSession(session_options, logging_manager) {
  }
};


}  // namespace test
}  // namespace onnxruntime
