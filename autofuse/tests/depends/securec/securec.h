// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Stub securec.h for open source LLT builds.
// Provides minimal secure C function stubs needed by test code.

#pragma once

#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _ERRNO_T_DEFINED
#define _ERRNO_T_DEFINED
typedef int errno_t;
#endif

#define SECUREC_INLINE static inline
#define EOK 0
#define SECUREC_MEM_MAX_LEN ((size_t)(-1) >> 1)

SECUREC_INLINE errno_t memcpy_s(void *dest, size_t destMax, const void *src, size_t count) {
  if (dest == NULL || src == NULL || destMax < count) {
    return -1;
  }
  memcpy(dest, src, count);
  return EOK;
}

SECUREC_INLINE errno_t memset_s(void *dest, size_t destMax, int c, size_t count) {
  if (dest == NULL || destMax < count) {
    return -1;
  }
  memset(dest, c, count);
  return EOK;
}

SECUREC_INLINE errno_t strcpy_s(char *strDest, size_t destMax, const char *strSrc) {
  if (strDest == NULL || strSrc == NULL || destMax == 0) {
    return -1;
  }
  size_t srcLen = strlen(strSrc);
  if (srcLen + 1 > destMax) {
    return -1;
  }
  memcpy(strDest, strSrc, srcLen + 1);
  return EOK;
}

SECUREC_INLINE errno_t strncpy_s(char *strDest, size_t destMax, const char *strSrc, size_t count) {
  if (strDest == NULL || strSrc == NULL || destMax == 0) {
    return -1;
  }
  size_t copyLen = count < destMax - 1 ? count : destMax - 1;
  strncpy(strDest, strSrc, copyLen);
  strDest[copyLen] = '\0';
  return EOK;
}

SECUREC_INLINE errno_t strcat_s(char *strDest, size_t destMax, const char *strSrc) {
  if (strDest == NULL || strSrc == NULL || destMax == 0) {
    return -1;
  }
  size_t destLen = strlen(strDest);
  size_t srcLen = strlen(strSrc);
  if (destLen + srcLen + 1 > destMax) {
    return -1;
  }
  memcpy(strDest + destLen, strSrc, srcLen + 1);
  return EOK;
}

SECUREC_INLINE int vsnprintf_s(char *dest, size_t destMax, size_t count, const char *fmt, va_list arglist) {
  if (dest == NULL || fmt == NULL || destMax == 0) {
    return -1;
  }
  (void)count;
  return vsnprintf(dest, destMax, fmt, arglist);
}

SECUREC_INLINE int snprintf_s(char *dest, size_t destMax, size_t count, const char *fmt, ...) {
  if (dest == NULL || fmt == NULL || destMax == 0) {
    return -1;
  }
  (void)count;
  va_list arglist;
  va_start(arglist, fmt);
  int ret = vsnprintf(dest, destMax, fmt, arglist);
  va_end(arglist);
  return ret;
}

#ifdef __cplusplus
}

// C++ variadic template version of sscanf_s (stub - delegates to sscanf)
#include <cstdio>
template <typename... Args>
SECUREC_INLINE int sscanf_s(const char *str, const char *format, Args... args) {
  return sscanf(str, format, args...);
}

#else
// C version of sscanf_s
SECUREC_INLINE int sscanf_s(const char *str, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int ret = vsscanf(str, format, args);
  va_end(args);
  return ret;
}
#endif
