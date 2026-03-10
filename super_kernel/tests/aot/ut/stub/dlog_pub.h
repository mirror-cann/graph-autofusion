/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES, OR KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Module ID for logging
typedef enum {
    ASCENDCKERNEL = 0,
    ASCENDSLOG = 1,
} AscendcModule;

// Log level definitions
typedef enum {
    DLOG_DEBUG = 0,
    DLOG_INFO = 1,
    DLOG_WARN = 2,
    DLOG_ERROR = 3,
    DLOG_FATAL = 4,
} DlogLevel;

// Stub logging functions using printf
static inline void dlog_debug(int32_t moduleId, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[DEBUG] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

static inline void dlog_info(int32_t moduleId, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[INFO] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

static inline void dlog_warn(int32_t moduleId, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[WARN] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

static inline void dlog_error(int32_t moduleId, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[ERROR] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

static inline void dlog_fatal(int32_t moduleId, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[FATAL] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

#ifdef __cplusplus
}
#endif
