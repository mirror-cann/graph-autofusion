/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GRAPH_TYPES_AF_H_
#define INC_EXTERNAL_GRAPH_TYPES_AF_H_

#include "graph/types.h"

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

#endif  // INC_EXTERNAL_GRAPH_TYPES_AF_H_
