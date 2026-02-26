/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "DevContext.h"
#include <vector>
#include <cassert>
#include "tensor_list.h"

std::vector<std::pair<int,int>> network__memory_malloc(DevContext &devObj);
void launch_network(DevContext &devObj, const std::vector<std::pair<int,int>>& layer_params_idx_size, int repeat_cnt);

void gen_rms_kernel_func(DevContext &devObj);
void gen_grouped_matmul_func(DevContext &devObj);
void gen_weight_quant_batch_matmul_v2_func(DevContext &devObj);
void gen_matmul_add_func(DevContext &devObj);
void gen_dequant_swiglu_quant_func(DevContext &devObj);

void ClearOpsKernelLaunch(uint32_t block_dim, void *stream);
void RmsNormKernel(uint32_t block_dim, void *stream, uint8_t *x, uint8_t *gamma, uint8_t *y, uint8_t *rstd, const RMSNormTilingData &tiling);
void GroupedMatmulKernel(uint32_t block_dim, void *stream,
                         uint8_t *x, uint8_t *weight, uint8_t *bias, uint8_t *scale,
                         uint8_t *offset, uint8_t *antiquantScale, uint8_t *antiquantOffset,
                         uint8_t *groupList, uint8_t *perTokenScale, uint8_t *y,
                         uint8_t *workspace, const GMMTilingData &tiling);
void WeightQuantBatchMatmulV2Kernel(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *weight, uint8_t *antiquantScale, uint8_t *antiquantOffset, uint8_t *quantScale, uint8_t *quantOffset, uint8_t *bias, uint8_t *y, uint8_t *workspace, const WeightQuantBatchMatmulV2MsdTilingData &tiling);
void MatmulAdd(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y);
void DequantSwiGluQuantDynamicKernel(uint32_t blockDim, void *stream,
                                         uint8_t *xGM, uint8_t *weightSscaleGM,
                                         uint8_t *activationScaleGM, uint8_t *biasGM,
                                         uint8_t *quantScaleGM, uint8_t *quantOffsetGM,
                                         uint8_t *groupIndex, uint8_t *yGM, uint8_t *scaleGM,
                                         uint8_t *workspace, const SwiGluTilingData &tiling);
void GroupedMatmulKernelV2(uint32_t block_dim, void *stream,
                         uint8_t *x, uint8_t *weight, uint8_t *bias, uint8_t *scale,
                         uint8_t *offset, uint8_t *antiquantScale, uint8_t *antiquantOffset,
                         uint8_t *groupList, uint8_t *perTokenScale, uint8_t *y,
                         uint8_t *workspace, const GMMTilingData &tiling);
void DequantSwiGluQuantDynamicKernel(uint32_t blockDim, void *stream,
                                         uint8_t *xGM, uint8_t *weightSscaleGM,
                                         uint8_t *activationScaleGM, uint8_t *biasGM,
                                         uint8_t *quantScaleGM, uint8_t *quantOffsetGM,
                                         uint8_t *groupIndex, uint8_t *yGM, uint8_t *scaleGM,
                                         uint8_t *workspace, const DequantSwigluQuantBaseTilingData &tiling);
void GroupedMatmulKernelV3(uint32_t block_dim, void *stream,
                         uint8_t *x, uint8_t *weight, uint8_t *bias, uint8_t *scale,
                         uint8_t *offset, uint8_t *antiquantScale, uint8_t *antiquantOffset,
                         uint8_t *groupList, uint8_t *perTokenScale, uint8_t *y,
                         uint8_t *workspace, const GMMTilingData &tiling);
void DynamicQuantKernel(uint32_t block_dim, void *stream,
    uint8_t *x, uint8_t *smooth_scales, uint8_t *group_index, uint8_t *y,
    uint8_t *scale, uint8_t *workSpace, const DynamicQuantTilingData& tiling);