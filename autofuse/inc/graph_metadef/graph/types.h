/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GRAPH_TYPES_H_
#define INC_EXTERNAL_GRAPH_TYPES_H_

#include <atomic>
#include <memory>
#include <vector>
#include "c_types.h"

// Primary definitions in namespace ge — required so that template instantiations
// (e.g. ge::GetTypeId<ge::DataType>) in pre-built CANN libraries link correctly.
namespace ge {
using char_t = char;
using float32_t = float;
using float64_t = double;
using vector_bit_t = std::vector<bool>;

static const int64_t SHAPE_RANGE_LOWER_LIMIT = 0;
static const int64_t UNKNOWN_DIM = -1;
static const int64_t UNKNOWN_DIM_NUM = -2;
#ifndef __NPU_DEVICE__
static const std::vector<int64_t> UNKNOWN_SHAPE = {-1};
static const std::vector<int64_t> UNKNOWN_RANK = {-2};
static const std::vector<int64_t> DUMMY_SHAPE = {-3};
#endif  // __NPU_DEVICE__
static constexpr int32_t kDataTypeSizeBitOffset = 1000;
static constexpr uint32_t kBitNumOfOneByte = 8U;
static constexpr uint32_t kBitThreeBytes = 24U;

#if defined(__GNUC__)
#ifndef GE_FUNC_HOST_VISIBILITY
#if defined(HOST_VISIBILITY)
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#endif  // GE_FUNC_HOST_VISIBILITY

#ifndef GE_FUNC_DEV_VISIBILITY
#if defined(DEV_VISIBILITY)
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif
#endif  // GE_FUNC_DEV_VISIBILITY

#ifndef WEAK_SYMBOL
#define WEAK_SYMBOL __attribute__((weak))
#endif

#ifndef FORMAT_PRINTF
#define FORMAT_PRINTF(format_idx, first_arg) __attribute__((format(printf, (format_idx), (first_arg))))
#endif
#else
#ifndef GE_FUNC_HOST_VISIBILITY
#define GE_FUNC_HOST_VISIBILITY
#endif

#ifndef GE_FUNC_DEV_VISIBILITY
#define GE_FUNC_DEV_VISIBILITY
#endif

#ifndef WEAK_SYMBOL
#define WEAK_SYMBOL
#endif

#ifndef FORMAT_PRINTF
#define FORMAT_PRINTF(format_idx, first_arg)
#endif
#endif  // defined(__GNUC__)

enum DataType {
  DT_FLOAT = ::C_DT_FLOAT,
  DT_FLOAT16 = ::C_DT_FLOAT16,
  DT_INT8 = ::C_DT_INT8,
  DT_INT32 = ::C_DT_INT32,
  DT_UINT8 = ::C_DT_UINT8,
  DT_INT16 = ::C_DT_INT16,
  DT_UINT16 = ::C_DT_UINT16,
  DT_UINT32 = ::C_DT_UINT32,
  DT_INT64 = ::C_DT_INT64,
  DT_UINT64 = ::C_DT_UINT64,
  DT_DOUBLE = ::C_DT_DOUBLE,
  DT_BOOL = ::C_DT_BOOL,
  DT_STRING = ::C_DT_STRING,
  DT_DUAL_SUB_INT8 = ::C_DT_DUAL_SUB_INT8,
  DT_DUAL_SUB_UINT8 = ::C_DT_DUAL_SUB_UINT8,
  DT_COMPLEX64 = ::C_DT_COMPLEX64,
  DT_COMPLEX128 = ::C_DT_COMPLEX128,
  DT_QINT8 = ::C_DT_QINT8,
  DT_QINT16 = ::C_DT_QINT16,
  DT_QINT32 = ::C_DT_QINT32,
  DT_QUINT8 = ::C_DT_QUINT8,
  DT_QUINT16 = ::C_DT_QUINT16,
  DT_RESOURCE = ::C_DT_RESOURCE,
  DT_STRING_REF = ::C_DT_STRING_REF,
  DT_DUAL = ::C_DT_DUAL,
  DT_VARIANT = ::C_DT_VARIANT,
  DT_BF16 = ::C_DT_BF16,
  DT_UNDEFINED = ::C_DT_UNDEFINED,
  DT_INT4 = ::C_DT_INT4,
  DT_UINT1 = ::C_DT_UINT1,
  DT_INT2 = ::C_DT_INT2,
  DT_UINT2 = ::C_DT_UINT2,
  DT_COMPLEX32 = ::C_DT_COMPLEX32,
  DT_HIFLOAT8 = ::C_DT_HIFLOAT8,
  DT_FLOAT8_E5M2 = ::C_DT_FLOAT8_E5M2,
  DT_FLOAT8_E4M3FN = ::C_DT_FLOAT8_E4M3FN,
  DT_FLOAT8_E8M0 = ::C_DT_FLOAT8_E8M0,
  DT_FLOAT6_E3M2 = ::C_DT_FLOAT6_E3M2,
  DT_FLOAT6_E2M3 = ::C_DT_FLOAT6_E2M3,
  DT_FLOAT4_E2M1 = ::C_DT_FLOAT4_E2M1,
  DT_FLOAT4_E1M2 = ::C_DT_FLOAT4_E1M2,
  DT_HIFLOAT4 = ::C_DT_HIFLOAT4,
  DT_MAX = ::C_DT_MAX,
};

struct StringHead {
  int64_t addr;
  int64_t len;
};

inline int GetSizeByDataType(DataType data_type) {
  static int data_type_size[DT_MAX] = {
      4,                           // DT_FLOAT
      2,                           // DT_FLOAT16
      1,                           // DT_INT8
      4,                           // DT_INT32
      1,                           // DT_UINT8
      -1,                          // reserved
      2,                           // DT_INT16
      2,                           // DT_UINT16
      4,                           // DT_UINT32
      8,                           // DT_INT64
      8,                           // DT_UINT64
      8,                           // DT_DOUBLE
      1,                           // DT_BOOL
      -1,                          // DT_STRING
      1,                           // DT_DUAL_SUB_INT8
      1,                           // DT_DUAL_SUB_UINT8
      8,                           // DT_COMPLEX64
      16,                          // DT_COMPLEX128
      1,                           // DT_QINT8
      2,                           // DT_QINT16
      4,                           // DT_QINT32
      1,                           // DT_QUINT8
      2,                           // DT_QUINT16
      8,                           // DT_RESOURCE
      -1,                          // DT_STRING_REF
      5,                           // DT_DUAL
      8,                           // DT_VARIANT
      2,                           // DT_BF16
      -1,                          // DT_UNDEFINED
      kDataTypeSizeBitOffset + 4,  // DT_INT4
      kDataTypeSizeBitOffset + 1,  // DT_UINT1
      kDataTypeSizeBitOffset + 2,  // DT_INT2
      kDataTypeSizeBitOffset + 2,  // DT_UINT2
      4,                           // DT_COMPLEX32
      1,                           // DT_HIFLOAT8
      1,                           // DT_FLOAT8_E5M2
      1,                           // DT_FLOAT8_E4M3FN
      1,                           // DT_FLOAT8_E8M0
      kDataTypeSizeBitOffset + 6,  // DT_FLOAT6_E3M2
      kDataTypeSizeBitOffset + 6,  // DT_FLOAT6_E2M3
      kDataTypeSizeBitOffset + 4,  // DT_FLOAT4_E2M1
      kDataTypeSizeBitOffset + 4,  // DT_FLOAT4_E1M2
      kDataTypeSizeBitOffset + 4,  // DT_HIFLOAT4
  };
  if ((data_type < 0) || (data_type >= DT_MAX)) {
    return -1;
  }
  return data_type_size[data_type];
}

int64_t GetSizeInBytes(int64_t element_count, DataType data_type);

enum Format {
  FORMAT_NCHW = ::C_FORMAT_NCHW,
  FORMAT_NHWC = ::C_FORMAT_NHWC,
  FORMAT_ND = ::C_FORMAT_ND,
  FORMAT_NC1HWC0 = ::C_FORMAT_NC1HWC0,
  FORMAT_FRACTAL_Z = ::C_FORMAT_FRACTAL_Z,
  FORMAT_NC1C0HWPAD = ::C_FORMAT_NC1C0HWPAD,
  FORMAT_NHWC1C0 = ::C_FORMAT_NHWC1C0,
  FORMAT_FSR_NCHW = ::C_FORMAT_FSR_NCHW,
  FORMAT_FRACTAL_DECONV = ::C_FORMAT_FRACTAL_DECONV,
  FORMAT_C1HWNC0 = ::C_FORMAT_C1HWNC0,
  FORMAT_FRACTAL_DECONV_TRANSPOSE = ::C_FORMAT_FRACTAL_DECONV_TRANSPOSE,
  FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = ::C_FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
  FORMAT_NC1HWC0_C04 = ::C_FORMAT_NC1HWC0_C04,
  FORMAT_FRACTAL_Z_C04 = ::C_FORMAT_FRACTAL_Z_C04,
  FORMAT_CHWN = ::C_FORMAT_CHWN,
  FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = ::C_FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS,
  FORMAT_HWCN = ::C_FORMAT_HWCN,
  FORMAT_NC1KHKWHWC0 = ::C_FORMAT_NC1KHKWHWC0,
  FORMAT_BN_WEIGHT = ::C_FORMAT_BN_WEIGHT,
  FORMAT_FILTER_HWCK = ::C_FORMAT_FILTER_HWCK,
  FORMAT_HASHTABLE_LOOKUP_LOOKUPS = ::C_FORMAT_HASHTABLE_LOOKUP_LOOKUPS,
  FORMAT_HASHTABLE_LOOKUP_KEYS = ::C_FORMAT_HASHTABLE_LOOKUP_KEYS,
  FORMAT_HASHTABLE_LOOKUP_VALUE = ::C_FORMAT_HASHTABLE_LOOKUP_VALUE,
  FORMAT_HASHTABLE_LOOKUP_OUTPUT = ::C_FORMAT_HASHTABLE_LOOKUP_OUTPUT,
  FORMAT_HASHTABLE_LOOKUP_HITS = ::C_FORMAT_HASHTABLE_LOOKUP_HITS,
  FORMAT_C1HWNCoC0 = ::C_FORMAT_C1HWNCoC0,
  FORMAT_MD = ::C_FORMAT_MD,
  FORMAT_NDHWC = ::C_FORMAT_NDHWC,
  FORMAT_FRACTAL_ZZ = ::C_FORMAT_FRACTAL_ZZ,
  FORMAT_FRACTAL_NZ = ::C_FORMAT_FRACTAL_NZ,
  FORMAT_NCDHW = ::C_FORMAT_NCDHW,
  FORMAT_DHWCN = ::C_FORMAT_DHWCN,
  FORMAT_NDC1HWC0 = ::C_FORMAT_NDC1HWC0,
  FORMAT_FRACTAL_Z_3D = ::C_FORMAT_FRACTAL_Z_3D,
  FORMAT_CN = ::C_FORMAT_CN,
  FORMAT_NC = ::C_FORMAT_NC,
  FORMAT_DHWNC = ::C_FORMAT_DHWNC,
  FORMAT_FRACTAL_Z_3D_TRANSPOSE = ::C_FORMAT_FRACTAL_Z_3D_TRANSPOSE,
  FORMAT_FRACTAL_ZN_LSTM = ::C_FORMAT_FRACTAL_ZN_LSTM,
  FORMAT_FRACTAL_Z_G = ::C_FORMAT_FRACTAL_Z_G,
  FORMAT_RESERVED = ::C_FORMAT_RESERVED,
  FORMAT_ALL = ::C_FORMAT_ALL,
  FORMAT_NULL = ::C_FORMAT_NULL,
  FORMAT_ND_RNN_BIAS = ::C_FORMAT_ND_RNN_BIAS,
  FORMAT_FRACTAL_ZN_RNN = ::C_FORMAT_FRACTAL_ZN_RNN,
  FORMAT_NYUV = ::C_FORMAT_NYUV,
  FORMAT_NYUV_A = ::C_FORMAT_NYUV_A,
  FORMAT_NCL = ::C_FORMAT_NCL,
  FORMAT_FRACTAL_Z_WINO = ::C_FORMAT_FRACTAL_Z_WINO,
  FORMAT_C1HWC0 = ::C_FORMAT_C1HWC0,
  FORMAT_FRACTAL_NZ_C0_16 = ::C_FORMAT_FRACTAL_NZ_C0_16,
  FORMAT_FRACTAL_NZ_C0_32 = ::C_FORMAT_FRACTAL_NZ_C0_32,
  FORMAT_FRACTAL_NZ_C0_2 = ::C_FORMAT_FRACTAL_NZ_C0_2,
  FORMAT_FRACTAL_NZ_C0_4 = ::C_FORMAT_FRACTAL_NZ_C0_4,
  FORMAT_FRACTAL_NZ_C0_8 = ::C_FORMAT_FRACTAL_NZ_C0_8,
  FORMAT_END = ::C_FORMAT_END,
  FORMAT_MAX = ::C_FORMAT_MAX,
};

inline int32_t GetFormatFromSub(int32_t primary_format, int32_t sub_format) {
  return static_cast<int32_t>((static_cast<uint32_t>(primary_format) & 0xffU) |
                              ((static_cast<uint32_t>(sub_format) & 0xffffU) << kBitNumOfOneByte));
}

inline int32_t GetFormatFromC0(int32_t format, int32_t c0_format) {
  return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffffffU) |
                              ((static_cast<uint32_t>(c0_format) & 0xfU) << kBitThreeBytes));
}

inline int32_t GetFormatFromSubAndC0(int32_t primary_format, int32_t sub_format, int32_t c0_format) {
  return static_cast<int32_t>((static_cast<uint32_t>(primary_format) & 0xffU) |
                              ((static_cast<uint32_t>(sub_format) & 0xffffU) << kBitNumOfOneByte) |
                              ((static_cast<uint32_t>(c0_format) & 0xfU) << kBitThreeBytes));
}

inline int32_t GetPrimaryFormat(int32_t format) {
  return static_cast<int32_t>(static_cast<uint32_t>(format) & 0xffU);
}

inline int32_t GetSubFormat(int32_t format) {
  return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffff00U) >> kBitNumOfOneByte);
}

inline bool HasSubFormat(int32_t format) {
  return GetSubFormat(format) > 0;
}

inline bool HasC0Format(int32_t format) {
  return ((static_cast<uint32_t>(format) & 0xf000000U) >> kBitThreeBytes) > 0;
}

inline int32_t GetC0Format(int32_t format) {
  return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xf000000U) >> kBitThreeBytes);
}

inline int64_t GetC0Value(int32_t format) {
  if (!HasC0Format(format)) {
    return -1;
  }
  return static_cast<int64_t>(
      1 << (static_cast<int32_t>((static_cast<uint32_t>(format) & 0xf000000U) >> kBitThreeBytes) - 1));
}

enum UnknowShapeOpType { DEPEND_IN_SHAPE = 1, DEPEND_CONST_VALUE = 2, DEPEND_SHAPE_RANGE = 3, DEPEND_COMPUTE = 4 };

struct TensorDescInfo {
  Format format_ = FORMAT_RESERVED;
  DataType dataType_ = DT_UNDEFINED;
};

enum DeviceType { NPU = 0, CPU = 1 };

enum Placement {
  kPlacementHost = 0,
  kPlacementDevice = 1,
  kPlacementEnd,
};

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const char_t *GetFormatName(Format format);

class TensorTypeImpl;
struct TensorType {
  explicit TensorType(DataType dt);
  TensorType(const std::initializer_list<DataType> &initial_types);

  static TensorType ALL() {
    return TensorType{DT_BOOL,   DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,    DT_INT32,
                      DT_INT64,  DT_INT8,       DT_QINT16,    DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8,   DT_RESOURCE,
                      DT_STRING, DT_UINT16,     DT_UINT32,    DT_UINT64, DT_UINT8, DT_BF16,    DT_COMPLEX32};
  }
  static TensorType QuantifiedType() {
    return TensorType{DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8};
  }
  static TensorType OrdinaryType() {
    return TensorType{DT_BOOL,  DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16, DT_INT32,
                      DT_INT64, DT_INT8,       DT_UINT16,    DT_UINT32, DT_UINT64, DT_UINT8,   DT_BF16,  DT_COMPLEX32};
  }
  static TensorType BasicType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,    DT_INT32,
                      DT_INT64,      DT_INT8,      DT_QINT16, DT_QINT32, DT_QINT8,   DT_QUINT16,  DT_QUINT8,
                      DT_UINT16,     DT_UINT32,    DT_UINT64, DT_UINT8,  DT_BF16,    DT_COMPLEX32};
  }
  static TensorType NumberType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                      DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT32, DT_QINT8,   DT_QUINT8,
                      DT_UINT16,     DT_UINT32,    DT_UINT64, DT_UINT8,  DT_BF16,    DT_COMPLEX32};
  }
  static TensorType RealNumberType() {
    return TensorType{DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,  DT_INT32, DT_INT64,
                      DT_INT8,   DT_UINT16, DT_UINT32,  DT_UINT64, DT_UINT8, DT_BF16};
  }
  static TensorType ComplexDataType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32};
  }
  static TensorType IntegerDataType() {
    return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
  }
  static TensorType SignedDataType() {
    return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8};
  }
  static TensorType UnsignedDataType() {
    return TensorType{DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
  }
  static TensorType FloatingDataType() {
    return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16};
  }
  static TensorType IndexNumberType() {
    return TensorType{DT_INT32, DT_INT64};
  }
  static TensorType UnaryDataType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_COMPLEX32};
  }
  static TensorType FLOAT() {
    return TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16};
  }

  std::shared_ptr<TensorTypeImpl> tensor_type_impl_;
};

struct ListTensorType {
  explicit ListTensorType(const TensorType &type) : tensor_type(type) {};
  TensorType tensor_type;
};

class Promote {
 public:
  friend class PromoteImpl;
  Promote(const std::initializer_list<const char *> &syms);
  std::vector<const char *> Syms() const;
  Promote(const Promote &other) = delete;
  Promote &operator=(const Promote &other) = delete;
  Promote(Promote &&other) noexcept;
  Promote &operator=(Promote &&other) noexcept;

 private:
  std::shared_ptr<void> data_;
};
}  // namespace ge

// af namespace aliases — autofusion source files use af:: types
namespace af {
using ge::char_t;
using ge::CPU;
using ge::DataType;
using ge::DEPEND_COMPUTE;
using ge::DEPEND_CONST_VALUE;
using ge::DEPEND_IN_SHAPE;
using ge::DEPEND_SHAPE_RANGE;
using ge::DeviceType;
using ge::float32_t;
using ge::float64_t;
using ge::Format;
using ge::GetC0Format;
using ge::GetC0Value;
using ge::GetFormatFromC0;
using ge::GetFormatFromSub;
using ge::GetFormatFromSubAndC0;
using ge::GetFormatName;
using ge::GetPrimaryFormat;
using ge::GetSizeByDataType;
using ge::GetSizeInBytes;
using ge::GetSubFormat;
using ge::HasC0Format;
using ge::HasSubFormat;
using ge::kPlacementDevice;
using ge::kPlacementEnd;
using ge::kPlacementHost;
using ge::ListTensorType;
using ge::NPU;
using ge::Placement;
using ge::Promote;
using ge::SHAPE_RANGE_LOWER_LIMIT;
using ge::StringHead;
using ge::TensorDescInfo;
using ge::TensorType;
using ge::TensorTypeImpl;
using ge::UNKNOWN_DIM;
using ge::UNKNOWN_DIM_NUM;
using ge::UnknowShapeOpType;
using ge::vector_bit_t;
#ifndef __NPU_DEVICE__
using ge::DUMMY_SHAPE;
using ge::UNKNOWN_RANK;
using ge::UNKNOWN_SHAPE;
#endif
using ge::DT_BF16;
using ge::DT_BOOL;
using ge::DT_COMPLEX128;
using ge::DT_COMPLEX32;
using ge::DT_COMPLEX64;
using ge::DT_DOUBLE;
using ge::DT_DUAL;
using ge::DT_DUAL_SUB_INT8;
using ge::DT_DUAL_SUB_UINT8;
using ge::DT_FLOAT;
using ge::DT_FLOAT16;
using ge::DT_FLOAT4_E1M2;
using ge::DT_FLOAT4_E2M1;
using ge::DT_FLOAT6_E2M3;
using ge::DT_FLOAT6_E3M2;
using ge::DT_FLOAT8_E4M3FN;
using ge::DT_FLOAT8_E5M2;
using ge::DT_FLOAT8_E8M0;
using ge::DT_HIFLOAT4;
using ge::DT_HIFLOAT8;
using ge::DT_INT16;
using ge::DT_INT2;
using ge::DT_INT32;
using ge::DT_INT4;
using ge::DT_INT64;
using ge::DT_INT8;
using ge::DT_MAX;
using ge::DT_QINT16;
using ge::DT_QINT32;
using ge::DT_QINT8;
using ge::DT_QUINT16;
using ge::DT_QUINT8;
using ge::DT_RESOURCE;
using ge::DT_STRING;
using ge::DT_STRING_REF;
using ge::DT_UINT1;
using ge::DT_UINT16;
using ge::DT_UINT2;
using ge::DT_UINT32;
using ge::DT_UINT64;
using ge::DT_UINT8;
using ge::DT_UNDEFINED;
using ge::DT_VARIANT;
using ge::FORMAT_ALL;
using ge::FORMAT_BN_WEIGHT;
using ge::FORMAT_C1HWC0;
using ge::FORMAT_C1HWNC0;
using ge::FORMAT_C1HWNCoC0;
using ge::FORMAT_CHWN;
using ge::FORMAT_CN;
using ge::FORMAT_DHWCN;
using ge::FORMAT_DHWNC;
using ge::FORMAT_END;
using ge::FORMAT_FILTER_HWCK;
using ge::FORMAT_FRACTAL_DECONV;
using ge::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS;
using ge::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS;
using ge::FORMAT_FRACTAL_DECONV_TRANSPOSE;
using ge::FORMAT_FRACTAL_NZ;
using ge::FORMAT_FRACTAL_NZ_C0_16;
using ge::FORMAT_FRACTAL_NZ_C0_2;
using ge::FORMAT_FRACTAL_NZ_C0_32;
using ge::FORMAT_FRACTAL_NZ_C0_4;
using ge::FORMAT_FRACTAL_NZ_C0_8;
using ge::FORMAT_FRACTAL_Z;
using ge::FORMAT_FRACTAL_Z_3D;
using ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE;
using ge::FORMAT_FRACTAL_Z_C04;
using ge::FORMAT_FRACTAL_Z_G;
using ge::FORMAT_FRACTAL_Z_WINO;
using ge::FORMAT_FRACTAL_ZN_LSTM;
using ge::FORMAT_FRACTAL_ZN_RNN;
using ge::FORMAT_FRACTAL_ZZ;
using ge::FORMAT_FSR_NCHW;
using ge::FORMAT_HASHTABLE_LOOKUP_HITS;
using ge::FORMAT_HASHTABLE_LOOKUP_KEYS;
using ge::FORMAT_HASHTABLE_LOOKUP_LOOKUPS;
using ge::FORMAT_HASHTABLE_LOOKUP_OUTPUT;
using ge::FORMAT_HASHTABLE_LOOKUP_VALUE;
using ge::FORMAT_HWCN;
using ge::FORMAT_MAX;
using ge::FORMAT_MD;
using ge::FORMAT_NC;
using ge::FORMAT_NC1C0HWPAD;
using ge::FORMAT_NC1HWC0;
using ge::FORMAT_NC1HWC0_C04;
using ge::FORMAT_NC1KHKWHWC0;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NCHW;
using ge::FORMAT_NCL;
using ge::FORMAT_ND;
using ge::FORMAT_ND_RNN_BIAS;
using ge::FORMAT_NDC1HWC0;
using ge::FORMAT_NDHWC;
using ge::FORMAT_NHWC;
using ge::FORMAT_NHWC1C0;
using ge::FORMAT_NULL;
using ge::FORMAT_NYUV;
using ge::FORMAT_NYUV_A;
using ge::FORMAT_RESERVED;
using ge::kBitNumOfOneByte;
using ge::kBitThreeBytes;
using ge::kDataTypeSizeBitOffset;
}  // namespace af

namespace domi {
enum class ImplyType : unsigned int {
  BUILDIN = 0,
  TVM,
  CUSTOM,
  AI_CPU,
  CCE,
  GELOCAL,
  INVALID = 0xFFFFFFFF,
};
using char_t = ge::char_t;
using float32_t = ge::float32_t;
using float64_t = ge::float64_t;
}  // namespace domi

#endif  // INC_EXTERNAL_GRAPH_TYPES_H_
