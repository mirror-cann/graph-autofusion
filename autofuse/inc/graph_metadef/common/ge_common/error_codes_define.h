/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Locally maintained copy of CANN error_codes_define.h with namespace ge -> af.

#ifndef PKG_INC_COMMON_GE_COMMON_ERROR_CODES_DEFINE_H_
#define PKG_INC_COMMON_GE_COMMON_ERROR_CODES_DEFINE_H_

#include <map>
#include <string>
#include "ge_common_af/ge_api_error_codes.h"

#define GE_ERRORNO_COMMON(name, value, desc)                                                                   \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::COMMON_MODULE, name, (value), (desc))
#define GE_ERRORNO_CLIENT(name, value, desc)                                                                   \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::CLIENT_MODULE, name, (value), (desc))
#define GE_ERRORNO_INIT(name, value, desc)                                                                     \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::INIT_MODULE, name, (value), (desc))
#define GE_ERRORNO_SESSION(name, value, desc)                                                                  \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::SESSION_MODULE, name, (value), (desc))
#define GE_ERRORNO_GRAPH(name, value, desc)                                                                    \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::GRAPH_MODULE, name, (value), (desc))
#define GE_ERRORNO_ENGINE(name, value, desc)                                                                   \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::ENGINE_MODULE, name, (value), (desc))
#define GE_ERRORNO_OPS(name, value, desc)                                                                      \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::OPS_MODULE, name, (value), (desc))
#define GE_ERRORNO_PLUGIN(name, value, desc)                                                                   \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::PLUGIN_MODULE, name, (value), (desc))
#define GE_ERRORNO_RUNTIME(name, value, desc)                                                                  \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::RUNTIME_MODULE, name, (value), (desc))
#define GE_ERRORNO_EXECUTOR(name, value, desc)                                                                   \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_DEVICE, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::EXECUTOR_MODULE, name, (value), (desc))
#define GE_ERRORNO_GENERATOR(name, value, desc)                                                                \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::GENERATOR_MODULE, name, (value), (desc))

#define LLM_ERRORNO_COMMON(name, value, desc)                                                                  \
  AF_GE_ERRORNO(af::InnLogRuntime::RT_HOST, af::InnErrorCodeType::ERROR_CODE, af::InnErrorLevel::COMMON_LEVEL, \
                af::InnSystemIdType::SYSID_GE, af::InnSubModuleId::LLM_ENGINE_MODULE, name, (value), (desc))

#define GE_GET_ERRORNO_STR(value) af::StatusFactory::Instance()->GetErrDesc(value)

#define RT_ERROR_TO_GE_STATUS(RT_ERROR) static_cast<af::Status>(RT_ERROR)

namespace af {
enum class InnSystemIdType : std::uint8_t { SYSID_GE = 8 };

enum class InnLogRuntime : std::uint8_t {
  RT_HOST = 0b01,
  RT_DEVICE = 0b10,
};

enum class InnSubModuleId : std::uint8_t {
  COMMON_MODULE = 0,
  CLIENT_MODULE = 1,
  INIT_MODULE = 2,
  SESSION_MODULE = 3,
  GRAPH_MODULE = 4,
  ENGINE_MODULE = 5,
  OPS_MODULE = 6,
  PLUGIN_MODULE = 7,
  RUNTIME_MODULE = 8,
  EXECUTOR_MODULE = 9,
  GENERATOR_MODULE = 10,
  LLM_ENGINE_MODULE = 11,
};

enum class InnErrorCodeType : std::uint8_t {
  ERROR_CODE = 0b01,
  EXCEPTION_CODE = 0b10,
};

enum class InnErrorLevel : std::uint8_t {
  COMMON_LEVEL = 0b000,
  SUGGESTION_LEVEL = 0b001,
  MINOR_LEVEL = 0b010,
  MAJOR_LEVEL = 0b011,
  CRITICAL_LEVEL = 0b100,
};

GE_ERRORNO_COMMON(PARAM_INVALID, 1, "Parameter invalid!");
GE_ERRORNO_COMMON(RT_FAILED, 3, "Failed to call runtime API!");
GE_ERRORNO_COMMON(INTERNAL_ERROR, 4, "Internal errors");
GE_ERRORNO_COMMON(GE_PLGMGR_FUNC_NOT_EXIST, 32, "Failed to find any function!");
GE_ERRORNO_COMMON(GE_PLGMGR_INVOKE_FAILED, 33, "Failed to invoke any function!");
GE_ERRORNO_GRAPH(NOT_CHANGED, 64, "The node of the graph no changed.");
}  // namespace af

namespace ge {
using InnSystemIdType = af::InnSystemIdType;
using InnLogRuntime = af::InnLogRuntime;
using InnSubModuleId = af::InnSubModuleId;
using InnErrorCodeType = af::InnErrorCodeType;
using InnErrorLevel = af::InnErrorLevel;
}  // namespace ge

#endif  // PKG_INC_COMMON_GE_COMMON_ERROR_CODES_DEFINE_H_
