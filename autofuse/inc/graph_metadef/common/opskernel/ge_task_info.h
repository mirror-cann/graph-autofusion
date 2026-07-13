/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_
#define INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_

#include <cstdint>
#include <string>
#include <vector>
#include "acl/acl_rt.h"
#include "runtime/rt.h"
#include "graph/op_desc.h"

namespace af {
struct DvppInfo {
  OpDescPtr op_desc;
  std::vector<void *> io_addrs;
  uint32_t sqe[16];
};

struct GETaskInfo {
  uint32_t id;
  uint16_t type;
  uint32_t streamID;
  void *stream;  // rtKernelLaunch input argument
  void *event;
  void *privateDef;
  uint32_t privateDefLen;
  void *opsKernelStorePtr;
  DvppInfo dvpp_info;
  bool needRefresh{false};
  std::vector<void *> rt_attached_streams;
};

struct HcomRemoteAccessAddrInfo {
  uint32_t remotetRankID;
  uint64_t remoteAddr;  // host embedding table address
  uint64_t localAddr;   // device HBM address
  uint64_t length;      // memory Length in Bytes
};

}  // namespace af
#endif  // INC_COMMON_OPSKERNEL_GE_TASK_INFO_H_
