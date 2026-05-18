/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pre_process/pre_process.h"
#include "pre_process/improve_precision.h"
#include "pre_process/scalar_broadcast_insert.h"
#include "ascgen_log.h"

namespace af {
namespace pre_process {

ge::Status PreProcess::Run(af::AscGraph &asc_graph) {
  // 对用户构图存在一些后端支持不了的场景做一些适配
  GELOGD("PreProcess::Run start, graph: %s.", asc_graph.GetName().c_str());

  GE_CHK_STATUS_RET(ImprovePrecisionForAscGraph(asc_graph), "Improve precision failed");

  GE_CHK_STATUS_RET(InsertBroadcastAfterScalarForAscGraph(asc_graph), "Insert broadcast after scalar failed");
  // 后续预处理步骤在此追加

  GELOGD("PreProcess::Run end, graph: %s.", asc_graph.GetName().c_str());
  return ge::SUCCESS;
}

}  // namespace pre_process
}  // namespace af
