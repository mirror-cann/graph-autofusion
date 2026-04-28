/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Bridge header: provides af:: namespace aliases for ge:: types from cann pkg.
 * Uses the same guard as the cann pkg types.h to avoid conflicts.
 */

#ifndef AUTOFUSE_EXTERNAL_GRAPH_TYPES_BRIDGE_H_
#define AUTOFUSE_EXTERNAL_GRAPH_TYPES_BRIDGE_H_

// Internal guard to prevent recursion when included via ./types.h or types.h
#ifndef AUTOFUSE_BRIDGE_TYPES_IN_PROGRESS_
#define AUTOFUSE_BRIDGE_TYPES_IN_PROGRESS_

// If the cann pkg header has not been included yet, include it now.
// If it has already been included elsewhere in this translation unit,
// still continue so the local af bridge declarations below are emitted.
#ifndef INC_EXTERNAL_GRAPH_TYPES_H_
#define AUTOFUSE_BRIDGE_INCLUDE_CANN_TYPES_
#endif
#ifdef AUTOFUSE_BRIDGE_INCLUDE_CANN_TYPES_
#include_next "graph/types.h"
#undef AUTOFUSE_BRIDGE_INCLUDE_CANN_TYPES_
#endif

#include "graph/ge_error_codes.h"
#include "graph/type/tensor_type_impl.h"

// Provide af:: aliases for code using af:: types
namespace af {
using ge::DataType;
using ge::Format;
using ge::FORMAT_ND;
using ge::DT_FLOAT;
using ge::DT_UNDEFINED;
using ge::FORMAT_END;
using ge::Promote;
using ge::char_t;
using ge::float32_t;
using ge::Placement;
using ge::kPlacementHost;
using ge::kPlacementDevice;
using ge::kPlacementEnd;
using ge::graphStatus;
using ge::DeviceType;
using ge::NPU;
using ge::UnknowShapeOpType;
using ge::FORMAT_RESERVED;
using ge::DT_FLOAT16;
using ge::DT_INT8;
using ge::DT_INT16;
using ge::DT_UINT16;
using ge::DT_UINT8;
using ge::DT_INT32;
using ge::DT_INT64;
using ge::DT_UINT32;
using ge::DT_UINT64;
using ge::DT_BOOL;
using ge::DT_DOUBLE;
using ge::DT_STRING;
using ge::DT_DUAL_SUB_INT8;
using ge::DT_DUAL_SUB_UINT8;
using ge::DT_COMPLEX64;
using ge::DT_COMPLEX128;
using ge::DT_QINT8;
using ge::DT_QINT16;
using ge::DT_QINT32;
using ge::DT_QUINT8;
using ge::DT_QUINT16;
using ge::DT_RESOURCE;
using ge::DT_STRING_REF;
using ge::DT_DUAL;
using ge::DT_BF16;
using ge::DT_MAX;
using ge::DT_COMPLEX32;
using ge::DT_VARIANT;
using ge::DT_INT4;
using ge::DT_UINT1;
using ge::DT_INT2;
using ge::DT_UINT2;
using ge::DT_HIFLOAT8;
using ge::DT_FLOAT8_E5M2;
using ge::DT_FLOAT8_E4M3FN;
using ge::DT_FLOAT8_E8M0;
using ge::DT_FLOAT6_E3M2;
using ge::DT_FLOAT6_E2M3;
using ge::DT_FLOAT4_E2M1;
using ge::DT_FLOAT4_E1M2;
using ge::UNKNOWN_DIM;
using ge::CPU;
using ge::GetSizeByDataType;
using ge::StringHead;
using ge::FORMAT_HASHTABLE_LOOKUP_LOOKUPS;
using ge::FORMAT_HASHTABLE_LOOKUP_KEYS;
using ge::FORMAT_HASHTABLE_LOOKUP_VALUE;
using ge::FORMAT_HASHTABLE_LOOKUP_OUTPUT;
using ge::FORMAT_HASHTABLE_LOOKUP_HITS;
using ge::GetFormatFromSub;
using ge::GetFormatFromSubAndC0;
using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;
using ge::FORMAT_NC1HWC0;
using ge::FORMAT_FRACTAL_Z;
using ge::FORMAT_NC1C0HWPAD;
using ge::FORMAT_NHWC1C0;
using ge::FORMAT_FSR_NCHW;
using ge::FORMAT_FRACTAL_DECONV;
using ge::FORMAT_C1HWNC0;
using ge::FORMAT_FRACTAL_DECONV_TRANSPOSE;
using ge::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS;
using ge::FORMAT_NC1HWC0_C04;
using ge::FORMAT_FRACTAL_Z_C04;
using ge::FORMAT_CHWN;
using ge::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS;
using ge::FORMAT_HWCN;
using ge::FORMAT_NC1KHKWHWC0;
using ge::FORMAT_BN_WEIGHT;
using ge::FORMAT_FILTER_HWCK;
using ge::FORMAT_C1HWNCoC0;
using ge::FORMAT_MD;
using ge::FORMAT_NDHWC;
using ge::FORMAT_FRACTAL_ZZ;
using ge::FORMAT_FRACTAL_NZ;
using ge::FORMAT_NCDHW;
using ge::FORMAT_DHWCN;
using ge::FORMAT_NDC1HWC0;
using ge::FORMAT_FRACTAL_Z_3D;
using ge::FORMAT_CN;
using ge::FORMAT_NC;
using ge::FORMAT_DHWNC;
using ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE;
using ge::FORMAT_FRACTAL_ZN_LSTM;
using ge::FORMAT_FRACTAL_Z_G;
using ge::FORMAT_ND_RNN_BIAS;
using ge::FORMAT_FRACTAL_ZN_RNN;
using ge::FORMAT_NYUV;
using ge::FORMAT_NYUV_A;
using ge::FORMAT_NCL;
using ge::FORMAT_FRACTAL_Z_WINO;
using ge::FORMAT_C1HWC0;
using ge::FORMAT_FRACTAL_NZ_C0_16;
using ge::FORMAT_FRACTAL_NZ_C0_32;
using ge::FORMAT_FRACTAL_NZ_C0_2;
using ge::FORMAT_FRACTAL_NZ_C0_4;
using ge::FORMAT_FRACTAL_NZ_C0_8;
using ge::FORMAT_MAX;
using ge::FORMAT_ALL;
using ge::FORMAT_NULL;
using ge::DUMMY_SHAPE;
using ge::UNKNOWN_DIM_NUM;
using ge::UNKNOWN_SHAPE;
using ge::UNKNOWN_RANK;
using ge::SHAPE_RANGE_LOWER_LIMIT;
using ge::kDataTypeSizeBitOffset;
using ge::kBitNumOfOneByte;

// af::TensorType is an independent struct (not ge::TensorType) to allow
// access to TensorTypeImpl internals used throughout af codebase.
struct TensorType {
  explicit TensorType(DataType dt);
  TensorType(const std::initializer_list<DataType> &initial_types);

  static TensorType ALL() {
    return TensorType{DT_BOOL, DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                      DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16,
                      DT_QUINT8, DT_RESOURCE, DT_STRING, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BF16};
  }
  static TensorType QuantifiedType() { return TensorType{DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8}; }
  static TensorType OrdinaryType() {
    return TensorType{DT_BOOL, DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                      DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BF16};
  }
  static TensorType BasicType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,
                      DT_INT32, DT_INT64, DT_INT8, DT_QINT16, DT_QINT32, DT_QINT8,
                      DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BF16};
  }
  static TensorType NumberType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                      DT_INT8, DT_QINT32, DT_QINT8, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BF16};
  }
  static TensorType RealNumberType() {
    return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64,
                      DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BF16};
  }
  static TensorType ComplexDataType() { return TensorType{DT_COMPLEX128, DT_COMPLEX64}; }
  static TensorType IntegerDataType() {
    return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
  }
  static TensorType SignedDataType() { return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8}; }
  static TensorType UnsignedDataType() { return TensorType{DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}; }
  static TensorType FloatingDataType() { return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16}; }
  static TensorType IndexNumberType() { return TensorType{DT_INT32, DT_INT64}; }
  static TensorType UnaryDataType() {
    return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_BF16};
  }
  static TensorType FLOAT() { return TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}; }

  std::shared_ptr<TensorTypeImpl> tensor_type_impl_;
};

struct ListTensorType {
  explicit ListTensorType(const TensorType &type) : tensor_type(type) {}
  TensorType tensor_type;
};

int64_t GetSizeInBytes(int64_t element_count, DataType data_type);

}  // namespace af

#endif  // AUTOFUSE_BRIDGE_TYPES_IN_PROGRESS_
#endif  // AUTOFUSE_EXTERNAL_GRAPH_TYPES_BRIDGE_H_
