/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_
#define INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_

#include <cstdint>

#ifndef CHAR_T_TYPEDEF
#define CHAR_T_TYPEDEF
using char_t = char;
#endif

#if(defined(HOST_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#if(defined(DEV_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif
#ifdef __GNUC__
#ifdef NO_METADEF_ABI_COMPATIABLE
#define ATTRIBUTED_DEPRECATED(replacement)
#define ATTRIBUTED_NOT_SUPPORT()
#else
#define ATTRIBUTED_DEPRECATED(replacement) __attribute__((deprecated("Please use " #replacement " instead.")))
#define ATTRIBUTED_NOT_SUPPORT() __attribute__((deprecated("The method will not be supported in the future.")))
#endif
#else
#ifdef NO_METADEF_ABI_COMPATIABLE
#define ATTRIBUTED_DEPRECATED(replacement)
#define ATTRIBUTED_NOT_SUPPORT()
#else
#define ATTRIBUTED_DEPRECATED(replacement) __declspec(deprecated("Please use " #replacement " instead."))
#define ATTRIBUTED_NOT_SUPPORT() __declspec(deprecated("The method will not be supported in the future."))
#endif
#endif

namespace af {
using graphStatus = uint32_t;
using Status = uint32_t;
const Status SUCCESS = 0U;
const Status FAILED = 0xFFFFFFFFU;
const graphStatus PARAM_INVALID = 1343225857U;
const graphStatus RT_FAILED = 1343291392U;
const graphStatus GRAPH_FAILED = 0xFFFFFFFF;
const graphStatus GRAPH_SUCCESS = 0;
const graphStatus GRAPH_NOT_CHANGED = 1343242304;
const graphStatus INTERNAL_ERROR = 1343225860U;
const graphStatus MEMALLOC_FAILED = 0x03000000U;
const graphStatus UNSUPPORTED = 0x03000064U;
const graphStatus OUT_OF_MEMORY = 0x03000065U;
const graphStatus PATH_INVALID = 0x03000066U;

const graphStatus GRAPH_PARAM_INVALID = 50331649;
const graphStatus GRAPH_NODE_WITHOUT_CONST_INPUT = 50331648;
const graphStatus GRAPH_NODE_NEED_REPASS = 50331647;
const graphStatus GRAPH_INVALID_IR_DEF = 50331646;
const graphStatus OP_WITHOUT_IR_DATATYPE_INFER_RULE = 50331645;
const graphStatus GRAPH_PARAM_OUT_OF_RANGE = 50331644;

const graphStatus GRAPH_MEM_OPERATE_FAILED = 50331539;
const graphStatus GRAPH_NULL_PTR = 50331538;
const graphStatus GRAPH_MEMCPY_FAILED = 50331537;
const graphStatus GRAPH_MEMSET_FAILED = 50331536;

const graphStatus GRAPH_MATH_CAL_FAILED = 50331429;
const graphStatus GRAPH_ADD_OVERFLOW = 50331428;
const graphStatus GRAPH_MUL_OVERFLOW = 50331427;
const graphStatus GRAPH_RoundUp_Overflow = 50331426;
const graphStatus GE_PLGMGR_FUNC_NOT_EXIST = 1343225888U;
const graphStatus GE_PLGMGR_INVOKE_FAILED = 1343225889U;
const graphStatus NOT_CHANGED = 1343242304U;
}  // namespace af

namespace ge {
using graphStatus = af::graphStatus;
using Status = af::Status;
#ifndef GE_ERRORNO_DEFINE
static const Status SUCCESS = af::SUCCESS;
static const Status FAILED = af::FAILED;
static const graphStatus PARAM_INVALID = af::PARAM_INVALID;
static const graphStatus RT_FAILED = af::RT_FAILED;
#endif
static const graphStatus GRAPH_FAILED = af::GRAPH_FAILED;
static const graphStatus GRAPH_SUCCESS = af::GRAPH_SUCCESS;
static const graphStatus GRAPH_NOT_CHANGED = af::GRAPH_NOT_CHANGED;
static const graphStatus GRAPH_PARAM_INVALID = af::GRAPH_PARAM_INVALID;
static const graphStatus GRAPH_NODE_WITHOUT_CONST_INPUT = af::GRAPH_NODE_WITHOUT_CONST_INPUT;
static const graphStatus GRAPH_NODE_NEED_REPASS = af::GRAPH_NODE_NEED_REPASS;
static const graphStatus GRAPH_INVALID_IR_DEF = af::GRAPH_INVALID_IR_DEF;
static const graphStatus OP_WITHOUT_IR_DATATYPE_INFER_RULE = af::OP_WITHOUT_IR_DATATYPE_INFER_RULE;
static const graphStatus GRAPH_PARAM_OUT_OF_RANGE = af::GRAPH_PARAM_OUT_OF_RANGE;
static const graphStatus GRAPH_MEM_OPERATE_FAILED = af::GRAPH_MEM_OPERATE_FAILED;
static const graphStatus GRAPH_NULL_PTR = af::GRAPH_NULL_PTR;
static const graphStatus GRAPH_MEMCPY_FAILED = af::GRAPH_MEMCPY_FAILED;
static const graphStatus GRAPH_MEMSET_FAILED = af::GRAPH_MEMSET_FAILED;
static const graphStatus GRAPH_MATH_CAL_FAILED = af::GRAPH_MATH_CAL_FAILED;
static const graphStatus GRAPH_ADD_OVERFLOW = af::GRAPH_ADD_OVERFLOW;
static const graphStatus GRAPH_MUL_OVERFLOW = af::GRAPH_MUL_OVERFLOW;
static const graphStatus GRAPH_RoundUp_Overflow = af::GRAPH_RoundUp_Overflow;
static const graphStatus INTERNAL_ERROR = af::INTERNAL_ERROR;
static const graphStatus MEMALLOC_FAILED = af::MEMALLOC_FAILED;
static const graphStatus UNSUPPORTED = af::UNSUPPORTED;
static const graphStatus OUT_OF_MEMORY = af::OUT_OF_MEMORY;
static const graphStatus PATH_INVALID = af::PATH_INVALID;
static const graphStatus GE_PLGMGR_FUNC_NOT_EXIST = af::GE_PLGMGR_FUNC_NOT_EXIST;
static const graphStatus GE_PLGMGR_INVOKE_FAILED = af::GE_PLGMGR_INVOKE_FAILED;
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_
