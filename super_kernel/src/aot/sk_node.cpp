/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_node.cpp
 * \brief
 */


#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include "sk_node.h"

namespace {
using FunMap = std::map<int, std::map<std::string, std::string>>;

FunMap DefaultMaps() {
    FunMap maps;
    maps[0] = {
        {"_Z8rms_normIDhDhEvPhS0_S0_S0_17RMSNormTilingData", "rms_norm_half_half_sk_mix_aiv"},
        {"_Z14grouped_matmulIDhDhDhDhDhDhEvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_13GMMTilingData", "grouped_matmul_half_sk"},
        {"_Z28weight_quant_batch_matmul_v2IDhaDh37WeightQuantBatchMatmulV2MsdTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_T2_", "weight_quant_batch_matmul_v2_half_int8_half_mc_sk"},
        {"_Z10matmul_addIhLh1ELh2EEvPhS0_", "matmul_add_uint8_sk"},
        {"_Z20dequant_swiglu_quantI16SwiGluTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_S1_T_", "dequant_swiglu_quant_dynamic_bf16_sk_mix_aiv"},
        {"clear_ops", "clear_ops_sk"},
        {"_Z20dequant_swiglu_quantI32DequantSwigluQuantBaseTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_S1_T_", "dequant_swiglu_quant_dynamic_int32_sk_mix_aiv"},
        {"_Z17grouped_matmul_v2IaiaiimEvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_13GMMTilingData", "grouped_matmul_v2_sk"},
        {"grouped_matmul_v3", "grouped_matmul_v3_sk"},
        {"dynamic_quant", "dynamic_quant_sk_mix_aiv"},
    };
    maps[1] = {
        {"_Z8rms_normIDhDhEvPhS0_S0_S0_17RMSNormTilingData", "rms_norm_half_half_sk1_mix_aiv"},
        {"_Z14grouped_matmulIDhDhDhDhDhDhEvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_13GMMTilingData", "grouped_matmul_half_sk1"},
        {"_Z28weight_quant_batch_matmul_v2IDhaDh37WeightQuantBatchMatmulV2MsdTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_T2_", "weight_quant_batch_matmul_v2_half_int8_half_mc_sk1"},
        {"_Z10matmul_addIhLh1ELh2EEvPhS0_", "matmul_add_uint8_sk1"},
        {"_Z20dequant_swiglu_quantI16SwiGluTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_S1_T_", "dequant_swiglu_quant_dynamic_bf16_sk1_mix_aiv"},
        {"clear_ops", "clear_ops_sk1"},
        {"_Z20dequant_swiglu_quantI32DequantSwigluQuantBaseTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_S1_T_", "dequant_swiglu_quant_dynamic_int32_sk1_mix_aiv"},
        {"_Z17grouped_matmul_v2IaiaiimEvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_13GMMTilingData", "grouped_matmul_v2_sk1"},
        {"grouped_matmul_v3", "grouped_matmul_v3_sk1"},
        {"dynamic_quant", "dynamic_quant_sk1_mix_aiv"},
    };
    maps[2] = {
        {"_Z8rms_normIDhDhEvPhS0_S0_S0_17RMSNormTilingData", "rms_norm_half_half_sk2_mix_aiv"},
        {"_Z14grouped_matmulIDhDhDhDhDhDhEvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_13GMMTilingData", "grouped_matmul_half_sk2"},
        {"_Z28weight_quant_batch_matmul_v2IDhaDh37WeightQuantBatchMatmulV2MsdTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_T2_", "weight_quant_batch_matmul_v2_half_int8_half_mc_sk2"},
        {"_Z10matmul_addIhLh1ELh2EEvPhS0_", "matmul_add_uint8_sk2"},
        {"_Z20dequant_swiglu_quantI16SwiGluTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_S1_T_", "dequant_swiglu_quant_dynamic_bf16_sk2_mix_aiv"},
        {"clear_ops", "clear_ops_sk2"},
        {"_Z20dequant_swiglu_quantI32DequantSwigluQuantBaseTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_S1_T_", "dequant_swiglu_quant_dynamic_int32_sk2_mix_aiv"},
        {"_Z17grouped_matmul_v2IaiaiimEvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_13GMMTilingData", "grouped_matmul_v2_sk2"},
        {"grouped_matmul_v3", "grouped_matmul_v3_sk2"},
        {"dynamic_quant", "dynamic_quant_sk2_mix_aiv"},
    };
    maps[3] = {
        {"_Z8rms_normIDhDhEvPhS0_S0_S0_17RMSNormTilingData", "rms_norm_half_half_sk3_mix_aiv"},
        {"_Z14grouped_matmulIDhDhDhDhDhDhEvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_13GMMTilingData", "grouped_matmul_half_sk3"},
        {"_Z28weight_quant_batch_matmul_v2IDhaDh37WeightQuantBatchMatmulV2MsdTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_T2_", "weight_quant_batch_matmul_v2_half_int8_half_mc_sk3"},
        {"_Z10matmul_addIhLh1ELh2EEvPhS0_", "matmul_add_uint8_sk3"},
        {"_Z20dequant_swiglu_quantI16SwiGluTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_S1_T_", "dequant_swiglu_quant_dynamic_bf16_sk3_mix_aiv"},
        {"clear_ops", "clear_ops_sk3"},
        {"_Z20dequant_swiglu_quantI32DequantSwigluQuantBaseTilingDataEvPhS1_S1_S1_S1_S1_S1_S1_S1_S1_T_", "dequant_swiglu_quant_dynamic_int32_sk3_mix_aiv"},
        {"_Z17grouped_matmul_v2IaiaiimEvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_13GMMTilingData", "grouped_matmul_v2_sk3"},
        {"grouped_matmul_v3", "grouped_matmul_v3_sk3"},
        {"dynamic_quant", "dynamic_quant_sk3_mix_aiv"},
    };
    return maps;
}

const FunMap &GetFunMaps() {
    static const FunMap maps = DefaultMaps();
    return maps;
}

std::string LookupSkName(int binIndex, const char *funcName) {
    if (!funcName) {
        SK_LOGE("invalid function name: funcName is null, binIndex=%d", binIndex);
        throw std::runtime_error("[sk error] invalid function name");
    }
    const auto &maps = GetFunMaps();
    auto binIt = maps.find(binIndex);
    if (binIt == maps.end()) {
        SK_LOGE("invalid binIndex: binIndex=%d, valid range=[0,3], funcName=%s", binIndex, funcName);
        throw std::runtime_error("[sk error] invalid binIdx");
    }
    auto nameIt = binIt->second.find(funcName);
    if (nameIt == binIt->second.end()) {
        SK_LOGE("unsupported function name: funcName=%s, binIndex=%d", funcName, binIndex);
        throw std::runtime_error("[sk error] unsupported function name : " + std::string(funcName));
    }
    return nameIt->second;
}

ResolvedFunctionInfo ResolveSkFunction(void *binHdl, const char *origName, const char *skName) {
    if (!binHdl || !origName || !skName) {
        SK_LOGE("resolve sk function invalid args: binHdl=%p, origName=%p, skName=%p", binHdl, origName, skName);
        throw std::runtime_error("[sk error] resolve sk function invalid args");
    }
    ResolvedFunctionInfo info{};
    aclrtFuncHandle fhdl = nullptr;
    CHECK_ACL(aclrtBinaryGetFunction((aclrtBinHandle)binHdl, skName, &fhdl));
    void *addr[2] = {nullptr, nullptr};
    CHECK_ACL(aclrtGetFunctionAddr(fhdl, addr, addr + 1));
    info.funcAddr[0] = (uint64_t)addr[0];
    info.funcAddr[1] = (uint64_t)addr[1];
    info.funcHdl = fhdl;
    return info;
}

SkKernelType NormalizeKernelType(uint32_t kernelType, const uint32_t taskRatio[2]) {
    switch (kernelType) {
        case K_TYPE_AIC:
        case K_TYPE_AIC_ROLLBACK:
            return SkKernelType::AIC_ONLY;
        case K_TYPE_AIV:
        case K_TYPE_AIV_ROLLBACK:
        case K_TYPE_MIX_AIV_MAIN:
            return SkKernelType::AIV_ONLY;
        case K_TYPE_MIX_AIC_MAIN:
            if (taskRatio[1] == 0) {
                return SkKernelType::AIC_ONLY;
            }
            if (taskRatio[1] == 1) {
                return SkKernelType::MIX_AIC_1_1;
            }
            if (taskRatio[1] == 2) {
                return SkKernelType::MIX_AIC_1_2;
            }
            break;
        default:
            break;
    }
    return SkKernelType::DEFAULT;
}
} // namespace

bool SuperKernelBaseNode::InitNode() {
    if (originTask == nullptr) {
        SK_LOGE("Origin task is null for nodeIdxInStream %lu in streamIdxInGraph %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelKernelNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    if (aclrtTaskGetKernelParams(*originTask, &kernelParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to get kernel params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    if (kernelParams.taskGrp != nullptr) {
        SK_LOGI("Kernel task group is not null for task %u in stream %u, which cannot be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
        return true;
    }
    SK_LOGI("Kernel node %s for task %u in stream %u can be fused in super kernel.", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
    isFusible = true;

    nodeInfos.kernelInfos.taskRatio[0] = kernelParams.sk_task_ratio[0];
    nodeInfos.kernelInfos.taskRatio[1] = kernelParams.sk_task_ratio[1];
    nodeInfos.kernelInfos.kernelType = NormalizeKernelType(kernelParams.sk_kernel_type, kernelParams.sk_task_ratio); // todo : 通过aclrtGetFunctionAttribute进行获取 参数选项：,ratio也是通过参数选项进行获取
    nodeInfos.kernelInfos.numBlocks = originTask->kernel.numBlocks;
    nodeInfos.kernelInfos.devArgs = originTask->kernel.devArgs;
    nodeInfos.kernelInfos.binHdl = reinterpret_cast<aclrtBinHandle>(kernelParams.binHandle);
    nodeInfos.kernelInfos.funcHdl = kernelParams.funcHandle;
    if (kernelParams.func_name != nullptr) {
        nodeInfos.kernelInfos.funcName = kernelParams.func_name;  // todo ： 需要调用aclrtGetFunctionName进行获取
    }
    if (!nodeInfos.kernelInfos.funcName.empty() && nodeInfos.kernelInfos.binHdl != nullptr) {
        for (size_t i = 0; i < kMaxSplitBinCount; ++i) {
            std::string skName = LookupSkName(static_cast<int>(i), nodeInfos.kernelInfos.funcName.c_str());
            nodeInfos.kernelInfos.resolvedFuncs[i] = ResolveSkFunction((void *)nodeInfos.kernelInfos.binHdl,
                                                              nodeInfos.kernelInfos.funcName.c_str(),
                                                              skName.c_str());
        }
    }
    return true;
}

bool SuperKernelKernelNode::InValidateNode() {
    if (!isFusible) {
        SK_LOGE("Kernel node %s for task %u in stream %u can not be fused in super kernel, which should not been invalidated.", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    SK_LOGI("Invalidating kernel node %s for task %u in stream %u, which will be fused in super kernel.", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
    kernelParams.flag = aclrtTaskFlag::ACL_RT_TASK_INVALID;
    if (aclrtTaskSetKernelParams(*originTask, &kernelParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate kernel node %s for task %u in stream %u", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelKernelNode::Update(const UpdateContext &ctx) {
    // 无参数 → invalid
    if (ctx.launchInfo == nullptr || ctx.skEntryFunc == nullptr) {
        return InValidateNode();
    }

    // 有参数 → 持久化到 RTS buffer
    aclrtArgsHandle ahdl = nullptr;
    aclrtParamHandle phdl = nullptr;
    size_t memSize = 0;
    size_t devArgsSize = 0;
    void *buf = nullptr;
    void *argsPtr = nullptr;

    const size_t MAX_HANDLE_MEM_SIZE = 1024 * 1024;  // 1MB 
    const size_t MAX_ARGS_MEM_SIZE = 256 * 1024 * 1024;  // 64MB 
    CHECK_ACL(aclrtKernelArgsGetHandleMemSize(ctx.skEntryFunc, &memSize)); 
    if (memSize == 0 || memSize > MAX_HANDLE_MEM_SIZE) { 
        SK_LOGE("invalid memSize: %zu", memSize); 
        return false; 
    } 
    ahdl = (aclrtArgsHandle)malloc(memSize);
    if (ahdl == nullptr) {
        SK_LOGE("malloc memSize failed");
        return false;
    }
    CHECK_ACL(aclrtKernelArgsGetMemSize(ctx.skEntryFunc, ctx.launchInfo->devArgs.get()->skHeader.totalSize, &devArgsSize));
    if (devArgsSize == 0 || devArgsSize > MAX_ARGS_MEM_SIZE) {
        SK_LOGE("invalid devArgsSize: %zu", devArgsSize);
        return false;
    }
    void *devArgs = nullptr;
    devArgs = malloc(devArgsSize);
    if (devArgs == nullptr) {
        SK_LOGE("malloc devArgsSize failed");
        return false;
    }
    CHECK_ACL(aclrtKernelArgsInitByUserMem(ctx.skEntryFunc, ahdl, devArgs, devArgsSize));

    CHECK_ACL(aclrtKernelArgsAppendPlaceHolder(ahdl, &phdl));
    CHECK_ACL(aclrtKernelArgsGetPlaceHolderBuffer(ahdl, phdl, ctx.launchInfo->devArgs.get()->skHeader.totalSize, (void**)&argsPtr));
    errno_t err = memcpy_s(argsPtr, ctx.launchInfo->devArgs.get()->skHeader.totalSize, ctx.launchInfo->devArgs.get(), ctx.launchInfo->devArgs.get()->skHeader.totalSize);
    if (err != 0) {
        SK_LOGE("memcpy_s failed");
        return false;
    } 
    CHECK_ACL(aclrtKernelArgsFinalize(ahdl));

    kernelParams.funcHandle = ctx.skEntryFunc;
    kernelParams.argsHandle = ahdl;
    kernelParams.numBlocks = ctx.launchInfo->entryInfo.numBlocks;
    kernelParams.flag = aclrtTaskFlag::ACL_RT_TASK_VALID;

    bool ret = (aclrtTaskSetKernelParams(*originTask, &kernelParams) == ACL_SUCCESS);
    if (!ret) {
        SK_LOGE("Failed to update kernel node %s for task %u in stream %u", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
    } else {
        SK_LOGI("Updated kernel node %s for task %u in stream %u with argsHandle", kernelParams.func_name, nodeIdxInStream, streamIdxInGraph);
    }

    return ret;
}

bool SuperKernelEventNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    if (aclrtTaskGetEventParams(*originTask, &eventParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to get event params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    nodeInfos.syncInfos.eventId = eventParams.eventId;
    nodeInfos.syncInfos.addrValue = eventParams.eventAddr;
    if (eventParams.eventType != aclrtEventType::ACL_RT_EVENT_MEMORY) {
         SK_LOGI("Event type is not memory based for task %u in stream %u, which cannot be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
         return true;
    }
    SK_LOGI("Event type of task %u in stream %u is memory based, which can be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
    isFusible = true;
    return true;
}

bool SuperKernelEventNode::InValidateNode() {
    if (!isFusible) {
        SK_LOGE("Event node with eventId %lu for task %u in stream %u can not be fused in super kernel, which should not been invalidated.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    SK_LOGI("Invalidating event node with eventId %lu for task %u in stream %u, which will be fused in super kernel.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
    eventParams.flag = aclrtTaskFlag::ACL_RT_TASK_INVALID;
    if (aclrtTaskSetEventParams(*originTask, &eventParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate event node with eventId %lu for task %u in stream %u", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelMemoryNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    if (aclrtTaskGetMemValueParams(*originTask, &memValueParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to get memory value params for task %u in stream %u", nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    nodeInfos.syncInfos.eventId = reinterpret_cast<uint64_t>(memValueParams.valueAddr);
    nodeInfos.syncInfos.addrValue = memValueParams.valueAddr;
    SK_LOGI("Memory value type is write for task %u in stream %u, which can be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
    isFusible = true;
    return true;
}

bool SuperKernelMemoryNode::InValidateNode() {
    if (!isFusible) {
        SK_LOGE("Memory node with eventId %lu for task %u in stream %u can not be fused in super kernel, which should not been invalidated.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    SK_LOGI("Invalidating memory node with eventId %lu for task %u in stream %u, which will be fused in super kernel.", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
    memValueParams.flag = aclrtTaskFlag::ACL_RT_TASK_INVALID;
    if (aclrtTaskSetMemValueParams(*originTask, &memValueParams) != ACL_SUCCESS) {
        SK_LOGE("Failed to invalidate memory node with eventId %lu for task %u in stream %u", nodeInfos.syncInfos.eventId, nodeIdxInStream, streamIdxInGraph);
        return false;
    }
    return true;
}

bool SuperKernelDefaultNode::InitNode() {
    if (!SuperKernelBaseNode::InitNode()) {
        return false;
    }
    SK_LOGI("Default task type for task %lu in stream %u, which cannot be fused in super kernel.", nodeIdxInStream, streamIdxInGraph);
    return true;
}

bool SuperKernelDefaultNode::InValidateNode() {
    SK_LOGE("Default task type for task %lu in stream %u should not be invalidated.", nodeIdxInStream, streamIdxInGraph);
    return false;
}
