/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Shadow header: redirects common/ge_common/debug/ge_log.h to our lightweight wrapper.
 * This prevents pkg_inc/common/ge_common/debug/ge_log.h from being pulled in,
 * which would cause macro redefinition errors.
 */

#ifndef COMMON_GE_COMMON_DEBUG_GE_LOG_H_
#define COMMON_GE_COMMON_DEBUG_GE_LOG_H_

// Delegate to our standalone lightweight ge_log.h wrapper.
#include "graph/debug/ge_log.h"

#endif  // COMMON_GE_COMMON_DEBUG_GE_LOG_H_
