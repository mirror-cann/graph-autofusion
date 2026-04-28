/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Provides af::TypeUtilsImpl by delegating to libmetadef.so symbols.
 * Implementation is in type_utils_impl.cc (compiled with ge:: types visible
 * before af:: shadow headers, to get correct symbol names).
 */
#ifndef AF_INC_BASE_UTILS_TYPE_UTILS_IMPL_H_
#define AF_INC_BASE_UTILS_TYPE_UTILS_IMPL_H_

#include "external/graph/ascend_string.h"
#include "graph/types.h"

namespace af {
class TypeUtilsImpl {
 public:
  static AscendString DataTypeToAscendString(const DataType data_type);
  static DataType AscendStringToDataType(const AscendString &str);
  static AscendString FormatToAscendString(const Format format);
  static Format AscendStringToFormat(const AscendString &str);
  static Format DataFormatToFormat(const AscendString &str);
  static bool GetDataTypeLength(const DataType data_type, uint32_t &length);
};
}  // namespace af
#endif  // AF_INC_BASE_UTILS_TYPE_UTILS_IMPL_H_
