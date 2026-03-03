/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_log.h
 * \brief
 */

#ifndef __SK_LOG_H__
#define __SK_LOG_H__

#include <csignal>
#include <string>
#include "dlog_pub.h"

#define ASCENDC_MODULE_NAME static_cast<int32_t>(ASCENDCKERNEL)

constexpr const char* GetFileName(const char* path)
{
    const char* file = path;
    while (*path != '\0') {
        if (*path++ == '/') {
            file = path;
        }
    }
    return file;
}

void ReportErrorMessageInner(const std::string &code, const char* fmt, ...);
template <typename ...Arguments>
void ReportErrorMessage(const char* fmt, Arguments &&... args)
{
    std::string errorCode = "EZ9999";
    return ReportErrorMessageInner(errorCode, fmt, std::forward<Arguments>(args)...);
}

#ifdef __cplusplus
extern "C" {
#endif

#define SK_LOGD(format, ...)                                                                         \
    do {                                                                                             \
      dlog_debug(ASCENDC_MODULE_NAME, "[SK][[%s:%s] " format "\n", GetFileName(__FILE__), __FUNCTION__, ##__VA_ARGS__);             \
    } while (0)

#define SK_LOGW(format, ...)                                                                         \
    do {                                                                                             \
      dlog_warn(ASCENDC_MODULE_NAME, "[SK][%s:%s] " format "\n", GetFileName(__FILE__), __FUNCTION__, ##__VA_ARGS__);          \
    } while (0)

#define SK_LOGI(format, ...)                                                                         \
    do {                                                                                             \
      dlog_info(ASCENDC_MODULE_NAME, "[SK][%s:%s] " format "\n", GetFileName(__FILE__), __FUNCTION__, ##__VA_ARGS__);              \
    } while (0)

#define SK_LOGE(format, ...)                                                           \
    do {                                                                                           \
        dlog_error(ASCENDC_MODULE_NAME, "[SK][%s:%s] " format "\n", GetFileName(__FILE__), __FUNCTION__, ##__VA_ARGS__);     \
    } while (0)

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            SK_LOGE("aclError: %d", __ret);                                                 \
        }                                                                                   \
    } while (0)

#define REPORT_ERROR_MESSAGE(...)              \
    do {                                       \
        ReportErrorMessage(__VA_ARGS__);       \
    } while (0)

#define SK_ASSERT_RETVAL(cond, ret)                               \
    do {                                                          \
        if (!(cond)) {                                            \
            SK_LOGE_WITH_REPORT("Assert %s failed", #cond);       \
            return (ret);                                         \
        }                                                         \
    } while (0)

#define SK_LOGE_WITH_REPORT(format, ...)                                                           \
    do {                                                                                           \
        dlog_error(ASCENDC_MODULE_NAME, "[SK][%s:%s] " format "\n", GetFileName(__FILE__), __FUNCTION__, ##__VA_ARGS__);     \
        REPORT_ERROR_MESSAGE(format, ##__VA_ARGS__);                                                \
    } while (0)
#ifdef __cplusplus
}
#endif

#endif // __SK_LOG_H__
