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
 * \file sk_log.cpp
 * \brief
 */

#include <vector>
#include "sk_log.h"
#include "securec.h"
#include "base/err_mgr.h"

constexpr size_t LIMIT_PREDEFINED_MESSAGE = 1024U;

void ReportErrorMessageInner(const std::string &code, const char* fmt, ...)
{
    std::vector<char> buf(LIMIT_PREDEFINED_MESSAGE, '\0');
    va_list argList;
    va_start(argList, fmt);
    auto ret = vsnprintf_s(buf.data(), LIMIT_PREDEFINED_MESSAGE, LIMIT_PREDEFINED_MESSAGE - 1U, fmt, argList);
    if (ret == -1) {
        SK_LOGW("Construct error message failed, maybe the length of error message exceed limits: %zu", 
            LIMIT_PREDEFINED_MESSAGE);
    }
    va_end(argList);
    const std::vector<const char*> msgKey = {"message"};
    const std::vector<const char*> msgValue = {buf.data()};
    REPORT_PREDEFINED_ERR_MSG(code.c_str(), msgKey, msgValue);
}

