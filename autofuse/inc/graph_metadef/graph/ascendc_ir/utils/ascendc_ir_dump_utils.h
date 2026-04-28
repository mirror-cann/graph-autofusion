/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_ASCENDC_IR_DUMP_UTILS_H
#define METADEF_CXX_ASCENDC_IR_DUMP_UTILS_H

#include <string>
#include <iostream>
#include <fstream>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/utils/type_utils.h"
namespace af {
class DumpAscirGraph {
 public:
  static std::string DumpGraph(AscGraph &graph);
  static void WriteOutToFile(const std::string &filename, AscGraph &graph);
 private:
  static std::stringstream &TilingKeyStr(std::stringstream &ss, AscGraph &graph);
  static std::stringstream &NameStr(std::stringstream &ss, AscGraph &graph);
  static std::stringstream &AllAxisStr(std::stringstream &ss, AscGraph &graph);
  static std::stringstream &AscNodeAttrStr(std::stringstream &ss, AscNodeAttr &attr);
  static std::stringstream &AscTensorAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &MemAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &MemQueueAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &MemBufAttrStr(std::stringstream &ss, AscTensorAttr *attr);
  static std::stringstream &MemOptAttrStr(std::stringstream &ss, const AscTensorAttr *attr);
  static std::stringstream &NodesStr(std::stringstream &ss, AscNodeVisitor &nodes);
  static std::string ApiTypeToString(ApiType type);
  static std::string ComputUnitToString(ComputeUnit unit);
  static std::string ComputeTypeToString(ComputeType type);
  static std::string AllocTypeToString(AllocType type);
  static std::string PositionToString(Position position);
  static std::string HardwareToString(MemHardware hardware);
};
} // namespace af

#endif  // METADEF_CXX_ASCENDC_IR_DUMP_UTILS_H