/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Shadow header: replaces external/ge_common/ge_api_error_codes.h.
 *
 * The system ge_api_error_codes.h defines SUCCESS/FAILED as constexpr via
 * GE_ERRORNO macros, which conflicts with the static const definitions in
 * our ge_error_codes.h. This shadow provides the same interface using only
 * our lightweight ge_error_codes.h, avoiding the conflict.
 *
 * We do NOT #include_next the system header because it pulls in ge_api_types.h
 * which transitively includes graph/tensor.h and causes further conflicts.
 */

#ifndef AUTOFUSE_EXTERNAL_GE_COMMON_GE_API_ERROR_CODES_SHADOW_H_
#define AUTOFUSE_EXTERNAL_GE_COMMON_GE_API_ERROR_CODES_SHADOW_H_

// Our ge_error_codes.h provides ge::SUCCESS, ge::FAILED, ge::GRAPH_SUCCESS,
// ge::GRAPH_FAILED, ge::PARAM_INVALID, ge::RT_FAILED and all graph status codes.
#include "graph/ge_error_codes.h"

// Define GE_ERRORNO_DEFINE as a no-op so that any code that checks
// #ifndef GE_ERRORNO_DEFINE does not try to redefine these constants.
#ifndef GE_ERRORNO_DEFINE
#define GE_ERRORNO_DEFINE(runtime, type, level, sysid, modid, name, value)  // no-op
#endif

#endif  // AUTOFUSE_EXTERNAL_GE_COMMON_GE_API_ERROR_CODES_SHADOW_H_
