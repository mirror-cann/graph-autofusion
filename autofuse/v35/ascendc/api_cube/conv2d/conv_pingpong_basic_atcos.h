/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CONV_PINGPONG_BASIC_ATCOS_H
#define CONV_PINGPONG_BASIC_ATCOS_H

#include "include/conv/block/block_scheduler.h"
#include "include/conv/prologue/block_prologue.h"

#ifdef CV_UB_FUSION
#include "include/conv/kernel/kernel_conv_forward_mix.h"
#include "include/conv/epilogue/block_epilogue_cv.h"
#else
#include "include/conv/kernel/kernel_conv_forward.h"
#include "include/conv/epilogue/block_epilogue_empty.h"
#endif

namespace Conv2DV2Advanced {

using namespace Atcos;
using namespace Atcos::Conv;

template<class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT, class BIAS_LAYOUT, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void ConvActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, 
#ifdef CV_UB_FUSION
    GM_ADDR cGM, GM_ADDR workspaceGM, const Conv2DTilingData& tilingData, AutoFusionVector::Params *param
#else
    GM_ADDR cGM, GM_ADDR workspaceGM, const Conv2DTilingData& tilingData
#endif
)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using CType = C_TYPE;
    using BiasType = BIAS_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using LayoutBias = BIAS_LAYOUT;

    using BlockScheduler = Block::IterateMFirst;

    using OutputOrder = order::OutputMMode;
    using BlockPrologue = Block::BlockPrologueEmpty;

    using BlockConvPolicy = Atcos::Conv::Img2ColConvMModePolicy;
    using BlockConv = Block::BlockConv<ConvShape, BlockConvPolicy, AType, BType, CType, BiasType, LayoutA, LayoutB, LayoutC, LayoutBias>;

    ConvShape shape;
    TilingData2ConvShape(tilingData, shape);
    ConvDim dimArgs;
    TilingData2ConvDim(tilingData, dimArgs);

#ifdef CV_UB_FUSION
    using FusionOp = AutoFusionVector;

    using BlockEpilogue = Block::BlockEpilogueCV<ConvShape, CType, CType, FusionOp>;

    using ConvKernel = Kernel::KernelConv<ConvShape, BlockConv, BlockPrologue, BlockEpilogue, BlockScheduler, OutputOrder>;

    typename ConvKernel::BlockConvArguments convArgs = {
        aGM, bGM, cGM, biasGM
    };

    typename BlockEpilogue::Params epilogueParams;
    epilogueParams.fusionParams = *param;

    typename ConvKernel::Params params = {
        shape,
        dimArgs,
        convArgs,
        {},
        epilogueParams
    };

    ConvKernel conv;
    conv(params);
#else
    using BlockEpilogue = Block::BlockEpilogueEmpty;

    using ConvKernel = Kernel::KernelConv<ConvShape, BlockConv, BlockPrologue, BlockEpilogue, BlockScheduler, OutputOrder>;

    typename ConvKernel::BlockConvArguments convArgs = {
        aGM, bGM, cGM, biasGM
    };

    typename ConvKernel::Params parmas = {
        shape,
        dimArgs,
        convArgs,
        {},
        {}
    };

    ConvKernel conv;
    conv(parmas);
#endif
}

}
#endif
