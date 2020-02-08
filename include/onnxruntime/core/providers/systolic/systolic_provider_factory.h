// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 * \param accelerator_mode 0 (default): CPU emulation | 1: Output stationary | 2: Weight Stationary
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Systolic, _In_ OrtSessionOptions* options, int use_arena, char accelerator_mode = 0)
ORT_ALL_ARGS_NONNULL;

#ifdef __cplusplus
}
#endif
