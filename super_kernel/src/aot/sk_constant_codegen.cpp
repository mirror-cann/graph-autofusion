/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_constant_codegen.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_common.h"  // 路径工具函数

#include <acl/acl.h>
#include <acl/acl_rt_compile.h>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <mutex>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

// 外部函数声明（来自 sk_entry 相关模块）
extern "C" aclrtBinHandle AscendGetEntryBinHandle();

// ==================== 常量代码模板 ====================

// 设备侧公共代码（枚举和结构体定义）
const char* SK_COMMON_CODE = R"(
#include <cstdint>
#include <cstddef>

enum class SkKernelType : uint8_t {
    DEFAULT = 0xFF,
    AIC_ONLY = 1,
    AIV_ONLY = 2,
    MIX_AIV_1_0 = 3,
    MIX_AIC_1_0 = 4,
    MIX_AIC_1_1 = 5,
    MIX_AIC_1_2 = 6,
};

enum class SkTaskType : uint8_t {
    TYPE_FUNC,
    TYPE_SYNC,
    TYPE_PRELOAD,
    TYPE_EVENT_NOTIFY,
    TYPE_EVENT_WAIT,
    TYPE_EVENT_RESET,
    TYPE_MAX,
};

enum class SkCoreSyncType : uint8_t {
    ALL_SYNC = 0,
    CROSS_SYNC_AIC_TO_AIC,
    CROSS_SYNC_AIV_TO_AIV,
    INTER_SYNC_SET_AIC_TO_AIV,
    INTER_SYNC_SET_AIV_TO_AIC,
    INTER_SYNC_WAIT_AIC_TO_AIV,
    INTER_SYNC_WAIT_AIV_TO_AIC,
    SYNC_NONE,
};

struct TaskInfo {
    uint32_t index;
    SkTaskType type;
    SkKernelType originType;
    uint8_t numBlocks;
    uint8_t entryCnt;
    uint64_t args;
    uint64_t entry[4];
    uint64_t debugOptions;
    uint64_t reserved;
};
)";

// 设备侧同步和事件处理代码
// 注意：kernel_operator.h 已包含 SkSystemArgs 定义
const char* SK_IMPL_CODE = R"(
#include "kernel_operator.h"

// 函数指针类型定义（使用 sk::SkSystemArgs，定义在 kernel_operator.h 中）
typedef void (*sk_sub_func)(const __gm__ void *param, const sk::SkSystemArgs *sysArgs);

namespace AscendC {

template<bool aic_flag>
__aicore__ inline void NotifyFunc(uint64_t param) {
    if constexpr(aic_flag) {
        if (get_block_idx() == 0) {
            __gm__ uint64_t *notifyLock = reinterpret_cast<__gm__ uint64_t *>(param);
            *notifyLock = 1;
            dcci(notifyLock, 0, 2);
        }
    } else {
        if (AscendC::GetBlockIdx() == 0) {
            __gm__ uint64_t *notifyLock = reinterpret_cast<__gm__ uint64_t *>(param);
            *notifyLock = 1;
            dcci(notifyLock, 0, 2);
        }
    }
}

template<bool aic_flag>
__aicore__ inline void WaitFunc(uint64_t param) {
    if constexpr(aic_flag) {
        if (get_block_idx() == 0) {
            __gm__ volatile uint64_t *waitLock = reinterpret_cast<__gm__ uint64_t *>(param);
            dcci(waitLock, 0, 2);
            while(*waitLock != 1) {
                dcci(waitLock, 0, 2);
            }
        }
    } else {
        if (AscendC::GetBlockIdx() == 0) {
            __gm__ volatile uint64_t *waitLock = reinterpret_cast<__gm__ uint64_t *>(param);
            dcci(waitLock, 0, 2);
            while(*waitLock != 1) {
                dcci(waitLock, 0, 2);
            }
        }
    }
}

template<bool aic_flag>
__aicore__ inline void ResetFunc(uint64_t param) {
    if constexpr(aic_flag) {
        if (get_block_idx() == 0) {
            __gm__ uint64_t *resetLock = reinterpret_cast<__gm__ uint64_t *>(param);
            *resetLock = 0;
            dcci(resetLock, 0, 2);
        }
    } else {
        if (AscendC::GetBlockIdx() == 0) {
            __gm__ uint64_t *resetLock = reinterpret_cast<__gm__ uint64_t *>(param);
            *resetLock = 0;
            dcci(resetLock, 0, 2);
        }
    }
}

template <uint8_t aic, uint8_t aiv>
__aicore__ inline void AutoCoreSyncImpl(SkCoreSyncType sync_type) {
    switch (sync_type) {
        case SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC:
            if ASCEND_IS_AIC {
                ffts_cross_core_sync(PIPE_FIX, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
                wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            }
            return;
        case SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV:
            if ASCEND_IS_AIV {
                ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
                wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            }
            return;
        case SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV:
            if ASCEND_IS_AIC {
                ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));
            }
            return;
        case SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC:
            if ASCEND_IS_AIV {
                ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
            }
            return;
        case SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV:
            if ASCEND_IS_AIV {
                wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
            }
            return;
        case SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC:
            if ASCEND_IS_AIC {
                wait_flag_dev(AscendC::SYNC_AIV_FLAG);
            }
            return;
        default:
            if constexpr (aic == 1 && aiv == 0) {
                ffts_cross_core_sync(PIPE_FIX, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
                wait_flag_dev(AscendC::SYNC_AIC_FLAG);
            } else if constexpr (aic == 0 && aiv == 1) {
                ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
                wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
            } else {
                AscendC::SyncAll<false>();
            }
            return;
    }
}

} // namespace AscendC
)";

// 辅助函数：将 TaskType 转换为字符串
const char* TaskTypeToEnumStr(SkTaskType type) {
    switch (type) {
        case SkTaskType::TYPE_FUNC: return "SkTaskType::TYPE_FUNC";
        case SkTaskType::TYPE_SYNC: return "SkTaskType::TYPE_SYNC";
        case SkTaskType::TYPE_PRELOAD: return "SkTaskType::TYPE_PRELOAD";
        case SkTaskType::TYPE_EVENT_NOTIFY: return "SkTaskType::TYPE_EVENT_NOTIFY";
        case SkTaskType::TYPE_EVENT_WAIT: return "SkTaskType::TYPE_EVENT_WAIT";
        case SkTaskType::TYPE_EVENT_RESET: return "SkTaskType::TYPE_EVENT_RESET";
        default: return "SkTaskType::TYPE_MAX";
    }
}

// 辅助函数：将 KernelType 转换为字符串
const char* KernelTypeToEnumStr(SkKernelType type) {
    switch (type) {
        case SkKernelType::AIC_ONLY: return "SkKernelType::AIC_ONLY";
        case SkKernelType::AIV_ONLY: return "SkKernelType::AIV_ONLY";
        case SkKernelType::MIX_AIV_1_0: return "SkKernelType::MIX_AIV_1_0";
        case SkKernelType::MIX_AIC_1_0: return "SkKernelType::MIX_AIC_1_0";
        case SkKernelType::MIX_AIC_1_1: return "SkKernelType::MIX_AIC_1_1";
        case SkKernelType::MIX_AIC_1_2: return "SkKernelType::MIX_AIC_1_2";
        default: return "SkKernelType::DEFAULT";
    }
}

// 辅助函数：将 uint64_t 转换为十六进制字符串
std::string Hex64ToStr(uint64_t val) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << val << "ULL";
    return oss.str();
}


// ==================== ConstantCodeGenerator 实现 ====================

std::string ConstantCodeGenerator::GenerateConstantTaskQue(
    const SkTask& task,
    const std::string& queueName)
{
    std::ostringstream code;
    const TaskQue* taskQue = task.GetTaskQue();
    if (taskQue == nullptr || taskQue->taskCnt == 0) {
        code << "// No " << queueName << " tasks\n";
        return code.str();
    }

    code << "struct TaskQue_" << queueName << " {\n";
    code << "    static constexpr uint32_t taskCnt = " << taskQue->taskCnt << ";\n";
    code << "    static constexpr uint32_t cap = " << taskQue->cap << ";\n";
    
    if (taskQue->taskCnt > 0) {
        code << "    static constexpr TaskInfo taskInfos[" << taskQue->taskCnt << "] = {\n";
        
        for (uint32_t i = 0; i < taskQue->taskCnt; i++) {
            const TaskInfo& info = taskQue->taskInfos[i];
            code << "        {";
            code << info.index << ", ";
            code << TaskTypeToEnumStr(info.type) << ", ";
            code << KernelTypeToEnumStr(info.originType) << ", ";
            code << static_cast<uint32_t>(info.numBlocks) << ", ";
            code << static_cast<uint32_t>(info.entryCnt) << ", ";
            code << Hex64ToStr(info.args) << ", ";
            
            // entry[4]
            code << "{";
            for (int j = 0; j < 4; j++) {
                code << Hex64ToStr(info.entry[j]);
                if (j < 3) code << ", ";
            }
            code << "}, ";
            
            code << Hex64ToStr(info.debugOptions) << ", ";
            code << Hex64ToStr(info.reserved);
            code << "}";
            
            if (i < taskQue->taskCnt - 1) code << ",";
            code << "\n";
        }
        
        code << "    };\n";
    }
    
    code << "};\n";
    return code.str();
}

/**
 * @brief 在编译期预计算同步配置值（复用 sk::SYNC_COMBINATION_TABLE）
 * @param preType 前一个任务的内核类型 (1-6)
 * @param curType 当前任务的内核类型 (1-6)
 * @return 同步配置值
 */
static uint16_t GetSyncCombinationValueStatic(int preType, int curType)
{
    // 复用 sk_common.h 中定义的 SYNC_COMBINATION_TABLE
    if (preType < 1 || preType > 6 || curType < 1 || curType > 6) {
        return sk::INVALID_SYNC_COMBINATION;
    }
    return sk::SYNC_COMBINATION_TABLE[preType - 1][curType - 1];
}

std::pair<int, int> ConstantCodeGenerator::GetKernelTypeParams(SkKernelType kernelType)
{
    switch (kernelType) {
        case SkKernelType::AIC_ONLY: return {1, 0};
        case SkKernelType::AIV_ONLY: return {0, 1};
        case SkKernelType::MIX_AIC_1_1: return {1, 1};
        case SkKernelType::MIX_AIC_1_2: return {1, 2};
        default: return {1, 1};
    }
}

/**
 * @brief 生成针对特定 split 的任务执行代码
 * @param taskQue 任务队列
 * @param taskIdx 任务索引
 * @param isAic 是否为 AIC 队列
 * @param preKernelType 前一个任务的内核类型
 * @param splitIdx split 索引 (0-3)，用于确定 entry 索引
 * @return 生成的代码字符串
 * 
 * 与 GenerateTaskExecution 不同，此函数直接使用编译期确定的 entry 索引，
 * 消除运行时的 get_coreid() 和取模运算。
 */
std::string ConstantCodeGenerator::GenerateTaskExecutionForSplit(
    const TaskQue* taskQue,
    size_t taskIdx,
    bool isAic,
    SkKernelType preKernelType,
    int splitIdx)
{
    std::ostringstream code;
    if (taskQue == nullptr || taskIdx >= taskQue->taskCnt) {
        return code.str();
    }
    
    const TaskInfo& task = taskQue->taskInfos[taskIdx];
    
    code << "        // Task[" << taskIdx << "]: " << TaskTypeToEnumStr(task.type) 
         << " (split=" << (splitIdx + 1) << ", entryIdx=" << (splitIdx % std::max(1, (int)task.entryCnt)) << ")\n";
    
    switch (task.type) {
        case SkTaskType::TYPE_PRELOAD: {
            code << "        {\n";
            code << "            auto blockId = AscendC::GetBlockIdx();\n";
            code << "            if (blockId < " << static_cast<uint32_t>(task.numBlocks) << ") {\n";
            
            // [SPLIT优化] 直接使用编译期确定的 entry 索引
            int entryIdx = (task.entryCnt > 0) ? (splitIdx % task.entryCnt) : 0;
            code << "                // [SPLIT] Preload entry[" << entryIdx << "]\n";
            code << "                constexpr uint64_t PRELOAD_ADDR = 0x" 
                 << std::hex << task.entry[entryIdx] << std::dec << "ULL;\n";
            code << "                preload((const void *)(PRELOAD_ADDR), " 
                 << static_cast<int64_t>(task.args) << "L);\n";
            
            // dc_preload 调用
            code << "                dc_preload(reinterpret_cast<__gm__ uint64_t*>(" << Hex64ToStr(task.reserved) << "), 0);\n";
            code << "                dc_preload(reinterpret_cast<__gm__ uint64_t*>(" << Hex64ToStr(task.reserved + 8) << "), 0);\n";
            code << "            }\n";
            code << "        }\n";
            break;
        }
        case SkTaskType::TYPE_FUNC: {
            code << "        {\n";
            code << "            auto blockId = AscendC::GetBlockIdx();\n";
            code << "            if (blockId < " << static_cast<uint32_t>(task.numBlocks) << ") {\n";
            
            // 预计算同步配置值
            uint16_t syncCfg = 0;
            if (preKernelType != SkKernelType::DEFAULT) {
                syncCfg = GetSyncCombinationValueStatic(
                    static_cast<int>(preKernelType), 
                    static_cast<int>(task.originType));
            }
            
            code << "                // [SPLIT] sysArgs: numBlocks=" << static_cast<uint32_t>(task.numBlocks)
                 << ", syncCfg=" << syncCfg << "\n";
            code << "                sk::SkSystemArgs sysArgs = {};\n";
            code << "                sysArgs.skBlockIdx = static_cast<uint16_t>(AscendC::GetBlockIdx());\n";
            code << "                sysArgs.skNumBlocks = " << static_cast<uint32_t>(task.numBlocks) << ";\n";
            code << "                sysArgs.skTaskSyncCfg = " << syncCfg << ";\n";
            
            // [SPLIT优化] 直接使用编译期确定的 entry 索引
            int entryIdx = (task.entryCnt > 0) ? (splitIdx % task.entryCnt) : 0;
            code << "                // [SPLIT] Func entry[" << entryIdx << "]\n";
            code << "                constexpr uint64_t FUNC_ADDR = 0x" 
                 << std::hex << task.entry[entryIdx] << std::dec << "ULL;\n";
            code << "                ((sk_sub_func)(FUNC_ADDR))"
                 << "(reinterpret_cast<const __gm__ void*>(" << Hex64ToStr(task.args) << "), &sysArgs);\n";
            code << "            }\n";
            code << "        }\n";
            break;
        }
        case SkTaskType::TYPE_SYNC: {
            code << "        AscendC::AutoCoreSyncImpl<aic, aiv>(static_cast<SkCoreSyncType>(" << task.args << "));\n";
            break;
        }
        case SkTaskType::TYPE_EVENT_NOTIFY: {
            code << "        if ASCEND_IS_AIC { AscendC::NotifyFunc<true>(" << Hex64ToStr(task.args) << "); }\n";
            code << "        if ASCEND_IS_AIV { AscendC::NotifyFunc<false>(" << Hex64ToStr(task.args) << "); }\n";
            break;
        }
        case SkTaskType::TYPE_EVENT_WAIT: {
            code << "        if ASCEND_IS_AIC { AscendC::WaitFunc<true>(" << Hex64ToStr(task.args) << "); }\n";
            code << "        if ASCEND_IS_AIV { AscendC::WaitFunc<false>(" << Hex64ToStr(task.args) << "); }\n";
            break;
        }
        case SkTaskType::TYPE_EVENT_RESET: {
            code << "        if ASCEND_IS_AIC { AscendC::ResetFunc<true>(" << Hex64ToStr(task.args) << "); }\n";
            code << "        if ASCEND_IS_AIV { AscendC::ResetFunc<false>(" << Hex64ToStr(task.args) << "); }\n";
            break;
        }
        default:
            break;
    }
    
    return code.str();
}

std::string ConstantCodeGenerator::GenerateSpecializedEntry(
    const SkTask& aicTask,
    const SkTask& aivTask,
    SkKernelType kernelType)
{
    std::ostringstream code;
    
    auto [aic, aiv] = GetKernelTypeParams(kernelType);
    const TaskQue* aicQue = aicTask.GetTaskQue();
    const TaskQue* aivQue = aivTask.GetTaskQue();
    
    SK_LOGI("[ConstantCodegen] GenerateSpecializedEntry: kernelType=%s, aic=%d, aiv=%d", 
            to_string(kernelType), aic, aiv);
    SK_LOGI("[ConstantCodegen] AIC queue: taskCnt=%u", aicQue ? aicQue->taskCnt : 0);
    SK_LOGI("[ConstantCodegen] AIV queue: taskCnt=%u", aivQue ? aivQue->taskCnt : 0);
    
    // 根据内核类型生成条件编译的常量 TaskQue
    code << "\n// ========== Constant TaskQue Definitions ==========\n";
    code << "#ifdef __DAV_CUBE__\n";
    code << GenerateConstantTaskQue(aicTask, "AIC");
    code << "#endif // __DAV_CUBE__\n\n";
    
    code << "#ifdef __DAV_VEC__\n";
    code << GenerateConstantTaskQue(aivTask, "AIV");
    code << "#endif // __DAV_VEC__\n\n";
    
    // ========== 生成 4 个 split 入口函数 ==========
    // 每个 split 函数处理 coreId % 4 == splitIdx 的情况
    for (int splitIdx = 0; splitIdx < 4; splitIdx++) {
        code << "// Split entry for coreId % 4 == " << splitIdx << "\n";
        code << "template <uint8_t aic, uint8_t aiv>\n";
        code << "__aicore__ __attribute__((aligned(512))) void sk_constant_entry_impl_split" << (splitIdx + 1) << "(void) {\n";
        
        // 生成针对该 split 的任务执行代码
        SkKernelType splitPreKernelType = SkKernelType::DEFAULT;
        
        // AIC 任务
        if (aicQue != nullptr && aicQue->taskCnt > 0) {
            code << "#ifdef __DAV_CUBE__\n";
            code << "    // AIC Queue Tasks (split " << (splitIdx + 1) << ")\n";
            for (uint32_t i = 0; i < aicQue->taskCnt; i++) {
                // 生成针对该 split 的任务执行代码
                code << GenerateTaskExecutionForSplit(aicQue, i, true, splitPreKernelType, splitIdx);
                if (aicQue->taskInfos[i].type == SkTaskType::TYPE_FUNC) {
                    splitPreKernelType = aicQue->taskInfos[i].originType;
                }
            }
            code << "#endif // __DAV_CUBE__\n";
        }
        
        // AIV 任务
        splitPreKernelType = SkKernelType::DEFAULT;
        if (aivQue != nullptr && aivQue->taskCnt > 0) {
            code << "#ifdef __DAV_VEC__\n";
            code << "    // AIV Queue Tasks (split " << (splitIdx + 1) << ")\n";
            for (uint32_t i = 0; i < aivQue->taskCnt; i++) {
                code << GenerateTaskExecutionForSplit(aivQue, i, false, splitPreKernelType, splitIdx);
                if (aivQue->taskInfos[i].type == SkTaskType::TYPE_FUNC) {
                    splitPreKernelType = aivQue->taskInfos[i].originType;
                }
            }
            code << "#endif // __DAV_VEC__\n";
        }
        
        code << "    pipe_barrier(PIPE_ALL);\n";
        code << "}\n\n";
    }
    
    // ========== 生成入口函数 ==========
    std::string entryFuncName = "sk_constant_entry_" + options_.skId;
    code << "// Entry point: dispatches to split functions based on coreId % 4\n";
    code << "extern \"C\" __global__ __attribute__((aligned(512))) __mix__(" << aic << ", " << aiv << ")\n";
    code << "void " << entryFuncName << "(void) {\n";
    code << "    uint8_t coreSplitIdx = (uint8_t)get_coreid() & 0x3;  // coreId % 4\n";
    code << "    switch (coreSplitIdx) {\n";
    code << "        case 0: sk_constant_entry_impl_split1<" << aic << ", " << aiv << ">(); break;\n";
    code << "        case 1: sk_constant_entry_impl_split2<" << aic << ", " << aiv << ">(); break;\n";
    code << "        case 2: sk_constant_entry_impl_split3<" << aic << ", " << aiv << ">(); break;\n";
    code << "        case 3: sk_constant_entry_impl_split4<" << aic << ", " << aiv << ">(); break;\n";
    code << "    }\n";
    code << "}\n";
    
    SK_LOGI("[ConstantCodegen] Generated entry function with 4 splits: %s", entryFuncName.c_str());
    
    return code.str();
}

std::string ConstantCodeGenerator::GenerateCombinedSource(
    const SkTask& aicTask,
    const SkTask& aivTask,
    SkKernelType kernelType)
{
    std::ostringstream source;
    
    // 1. 公共代码
    source << SK_COMMON_CODE << "\n";
    
    // 2. 实现代码
    source << SK_IMPL_CODE << "\n";
    
    // 3. 特化入口函数
    source << GenerateSpecializedEntry(aicTask, aivTask, kernelType);
    
    return source.str();
}

ConstantCodeGenResult ConstantCodeGenerator::Generate(
    const SkTask& aicTask,
    const SkTask& aivTask,
    const SkHeaderInfo& header)
{
    ConstantCodeGenResult result;
    
    if (!options_.enableConstantCodeGen) {
        SK_LOGI("Constant code generation is disabled");
        return result;
    }
    
    // 确定内核类型
    SkKernelType kernelType = SkKernelType::DEFAULT;
    const TaskQue* aicQue = aicTask.GetTaskQue();
    const TaskQue* aivQue = aivTask.GetTaskQue();
    
    uint32_t aicFuncCnt = aicTask.funcCnt;
    uint32_t aivFuncCnt = aivTask.funcCnt;
    
    if (aicFuncCnt == 0 && aivFuncCnt > 0) {
        kernelType = SkKernelType::AIV_ONLY;
    } else if (aicFuncCnt > 0 && aivFuncCnt == 0) {
        kernelType = SkKernelType::AIC_ONLY;
    } else if (aicFuncCnt > 0 && aivFuncCnt > 0) {
        if (aicTask.nodeType == SkKernelType::MIX_AIC_1_2 || aivTask.nodeType == SkKernelType::MIX_AIC_1_2) {
            kernelType = SkKernelType::MIX_AIC_1_2;
        } else {
            kernelType = SkKernelType::MIX_AIC_1_1;
        }
    }
    
    // 生成合并源码
    result.combinedSource = GenerateCombinedSource(aicTask, aivTask, kernelType);
    
    SK_LOGI("Generated constant source code, size=%zu bytes", result.combinedSource.size());
    
    return result;
}

ConstantCodeGenResult ConstantCodeGenerator::CompileAndResolve(
    const std::string& source,
    SkKernelType kernelType)
{
    ConstantCodeGenResult result;
    result.combinedSource = source;
    
    // 使用 aclrtc JIT 编译
    aclrtcProg prog = nullptr;
    aclError ret = aclrtcCreateProg(&prog, source.c_str(), "sk_constant.asc", 0, nullptr, nullptr);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to create aclrtc program: ret=%d", ret);
        return result;
    }
    
    // 编译选项
    const char* options[] = {
        "--npu-arch=dav-2201",
        "-O3"  // 启用优化
    };
    int numOptions = sizeof(options) / sizeof(options[0]);
    
    ret = aclrtcCompileProg(prog, numOptions, options);
    if (ret != ACL_SUCCESS) {
        // 获取编译日志
        size_t logSize = 0;
        aclrtcGetCompileLogSize(prog, &logSize);
        if (logSize > 0) {
            std::vector<char> logBuf(logSize);
            aclrtcGetCompileLog(prog, logBuf.data());
            SK_LOGE("aclrtc compilation failed:\n%s", logBuf.data());
        }
        aclrtcDestroyProg(&prog);
        return result;
    }
    
    // 获取编译后的二进制
    size_t binSize = 0;
    ret = aclrtcGetBinDataSize(prog, &binSize);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get binary size: ret=%d", ret);
        aclrtcDestroyProg(&prog);
        return result;
    }
    
    result.compiledBinary.resize(binSize);
    ret = aclrtcGetBinData(prog, reinterpret_cast<char*>(result.compiledBinary.data()));
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get binary data: ret=%d", ret);
        aclrtcDestroyProg(&prog);
        return result;
    }
    
    // 加载二进制并获取 funcHandle
    aclrtBinaryLoadOptions loadOpts;
    aclrtBinaryLoadOption opt;
    opt.type = ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC;
    opt.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    loadOpts.numOpt = 1;
    loadOpts.options = &opt;
    
    ret = aclrtBinaryLoadFromData(reinterpret_cast<void*>(result.compiledBinary.data()), 
                                   binSize, &loadOpts, &result.specializedBinHandle);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to load binary: ret=%d", ret);
        aclrtcDestroyProg(&prog);
        return result;
    }
    
    // 获取入口函数
    std::string entryFuncName = "sk_constant_entry_" + options_.skId;
    ret = aclrtBinaryGetFunction(result.specializedBinHandle, entryFuncName.c_str(), &result.specializedFuncHandle);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get function handle: ret=%d, funcName=%s", ret, entryFuncName.c_str());
        aclrtBinaryUnLoad(result.specializedBinHandle);
        result.specializedBinHandle = nullptr;
    }
    
    SK_LOGI("Successfully compiled and resolved funcHandle: binHandle=%p, funcHandle=%p",
            result.specializedBinHandle, result.specializedFuncHandle);
    
    aclrtcDestroyProg(&prog);
    return result;
}

// ==================== ConstantFuncHandleManager 实现 ====================

ConstantFuncHandleManager& ConstantFuncHandleManager::Instance()
{
    static ConstantFuncHandleManager instance;
    return instance;
}

void ConstantFuncHandleManager::RegisterFuncHandle(
    const std::string& skId,
    aclrtFuncHandle funcHandle,
    aclrtBinHandle binHandle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    HandlePair pair;
    pair.funcHandle = funcHandle;
    pair.binHandle = binHandle;
    handleMap_[skId] = pair;
    SK_LOGI("Registered constant funcHandle: skId=%s, funcHandle=%p, binHandle=%p",
            skId.c_str(), funcHandle, binHandle);
}

aclrtFuncHandle ConstantFuncHandleManager::GetFuncHandle(const std::string& skId) const
{
    auto it = handleMap_.find(skId);
    if (it != handleMap_.end()) {
        return it->second.funcHandle;
    }
    return nullptr;
}

aclrtBinHandle ConstantFuncHandleManager::GetBinHandle(const std::string& skId) const
{
    auto it = handleMap_.find(skId);
    if (it != handleMap_.end()) {
        return it->second.binHandle;
    }
    return nullptr;
}

bool ConstantFuncHandleManager::HasFuncHandle(const std::string& skId) const
{
    return handleMap_.find(skId) != handleMap_.end();
}

void ConstantFuncHandleManager::Clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : handleMap_) {
        if (pair.second.binHandle != nullptr) {
            aclrtBinaryUnLoad(pair.second.binHandle);
        }
    }
    handleMap_.clear();
}

/**
 * @brief 将生成的源码和二进制写入 sk_meta 目录
 * @param skId Super Kernel ID
 * @param sourceCode 生成的源码
 * @param binaryData 编译后的二进制数据
 * @param kernelType 内核类型
 * @param modelRI 模型 RI 指针
 * @return 是否写入成功
 */
bool DumpConstantCodegenFiles(
    const std::string& skId,
    const std::string& sourceCode,
    const std::vector<uint8_t>& binaryData,
    SkKernelType kernelType,
    aclmdlRI modelRI)
{
    // 获取 sk_meta 路径，使用真实的 modelRI
    std::string baseDir = CreateSkMetaDirectory(modelRI);
    if (baseDir.empty()) {
        SK_LOGE("[ConstantCodegen] Failed to create sk_meta directory");
        return false;
    }
    
    // 创建 constant_codegen 子目录
    std::string codegenDir = baseDir + "/constant_codegen";
    if (!CreateDirectoryRecursive(codegenDir)) {
        SK_LOGE("[ConstantCodegen] Failed to create constant_codegen directory: %s", codegenDir.c_str());
        return false;
    }
    
    // 文件名前缀
    std::string filePrefix = codegenDir + "/sk_" + skId + "_" + to_string(kernelType);
    
    // 1. 写入源码文件 (.asc)
    std::string sourceFile = filePrefix + ".asc";
    std::ofstream srcOut(sourceFile, std::ios::out);
    if (srcOut.is_open()) {
        srcOut << sourceCode;
        srcOut.close();
        SK_LOGI("[ConstantCodegen] Source code written to: %s (size=%zu bytes)", 
                sourceFile.c_str(), sourceCode.size());
    } else {
        SK_LOGE("[ConstantCodegen] Failed to write source file: %s", sourceFile.c_str());
        return false;
    }
    
    // 2. 写入二进制文件 (.bin)
    std::string binaryFile = filePrefix + ".bin";
    std::ofstream binOut(binaryFile, std::ios::binary);
    if (binOut.is_open()) {
        binOut.write(reinterpret_cast<const char*>(binaryData.data()), binaryData.size());
        binOut.close();
        SK_LOGI("[ConstantCodegen] Binary written to: %s (size=%zu bytes)", 
                binaryFile.c_str(), binaryData.size());
    } else {
        SK_LOGE("[ConstantCodegen] Failed to write binary file: %s", binaryFile.c_str());
        return false;
    }
    
    return true;
}

/**
 * @brief 尝试生成并使用常量化 funcHandle
 * 
 * 此函数是核心集成点，在 SkTaskBuilder::GenEntryInfo 中调用。
 * 如果常量化成功，返回新的 funcHandle；否则返回 nullptr，使用原有逻辑。
 * 
 * @param aicTask AIC 任务队列
 * @param aivTask AIV 任务队列
 * @param opts 选项管理器
 * @param modelRI 模型 RI 指针，用于生成 sk_meta 路径
 * @return std::pair<aclrtFuncHandle, SkKernelType> 
 *         first: 常量化 funcHandle（失败为 nullptr）
 *         second: 内核类型
 */
std::pair<aclrtFuncHandle, SkKernelType> TryGenerateConstantFuncHandle(
    const SkTask& aicTask,
    const SkTask& aivTask,
    SuperKernelOptionsManager& opts,
    aclmdlRI modelRI)
{
    SK_LOGI("[ConstantCodegen] Start constant codegen");
    
    // ========== 1. 检查是否启用常量化 ==========
    // 默认关闭，可通过环境变量 SK_CONSTANT=1 或选项显式开启
    // 首先检查环境变量
    bool enableConstant = false;
    const char* envConstant = std::getenv("SK_CONSTANT");
    if (envConstant != nullptr && std::string(envConstant) == "1") {
        enableConstant = true;
        SK_LOGI("[ConstantCodegen] Enabled by environment variable SK_CONSTANT=1");
    }
    
    // 然后检查选项（选项优先级高于环境变量）
    auto constantOpt = opts.GetOption(aclskOptionType::CONSTANT_CODEGEN);
    if (constantOpt != nullptr) {
        if (constantOpt->GetIntValue() == 1) {
            enableConstant = true;
            SK_LOGI("[ConstantCodegen] Enabled by CONSTANT_CODEGEN option");
        } else {
            enableConstant = false;  // 选项显式关闭时，覆盖环境变量
            SK_LOGI("[ConstantCodegen] Disabled by CONSTANT_CODEGEN option");
        }
    }
    
    if (!enableConstant) {
        SK_LOGI("[ConstantCodegen] Constant codegen is disabled (default). Set SK_CONSTANT=1 to enable.");
        return {nullptr, SkKernelType::DEFAULT};
    }
    
    SK_LOGI("[ConstantCodegen] Constant codegen is enabled");
    
    // ========== 1.5 检查基础入口函数二进制是否可用 ==========
    // 如果原始入口函数二进制不存在，则跳过常量化（保持原有测试逻辑）
    aclrtBinHandle entryBinHandle = AscendGetEntryBinHandle();
    if (entryBinHandle == nullptr) {
        SK_LOGI("[ConstantCodegen] Entry binHandle is null, skip constant codegen");
        return {nullptr, SkKernelType::DEFAULT};
    }
    
    // ========== 2. 分析任务队列信息 ==========
    const TaskQue* aicQue = aicTask.GetTaskQue();
    const TaskQue* aivQue = aivTask.GetTaskQue();
    
    SK_LOGI("[ConstantCodegen] Task analysis:");
    SK_LOGI("  - AIC: funcCnt=%u, numBlocks=%u, taskCnt=%u", 
            aicTask.funcCnt, aicTask.numBlocks, aicQue ? aicQue->taskCnt : 0);
    SK_LOGI("  - AIV: funcCnt=%u, numBlocks=%u, taskCnt=%u", 
            aivTask.funcCnt, aivTask.numBlocks, aivQue ? aivQue->taskCnt : 0);
    
    // 检查任务队列有效性
    if ((aicQue == nullptr || aicQue->taskCnt == 0) && 
        (aivQue == nullptr || aivQue->taskCnt == 0)) {
        SK_LOGE("[ConstantCodegen] Both AIC and AIV task queues are empty");
        return {nullptr, SkKernelType::DEFAULT};
    }
    
    // ========== 3. 确定内核类型 ==========
    SkKernelType kernelType = SkKernelType::MIX_AIC_1_1;
    if (aicTask.funcCnt == 0 && aivTask.funcCnt > 0) {
        kernelType = SkKernelType::AIV_ONLY;
    } else if (aicTask.funcCnt > 0 && aivTask.funcCnt == 0) {
        kernelType = SkKernelType::AIC_ONLY;
    } else if (aicTask.nodeType == SkKernelType::MIX_AIC_1_2 || aivTask.nodeType == SkKernelType::MIX_AIC_1_2) {
        kernelType = SkKernelType::MIX_AIC_1_2;
    }
    SK_LOGI("[ConstantCodegen] Kernel type determined: %s", to_string(kernelType));
    
    // ========== 4. 生成唯一 ID ==========
    static std::atomic<uint64_t> skIdCounter{0};
    std::string skId = std::to_string(skIdCounter.fetch_add(1));
    SK_LOGI("[ConstantCodegen] Generated skId: %s", skId.c_str());
    
    // ========== 5. 创建代码生成器 ==========
    ConstantCodeGenOptions codegenOpts;
    codegenOpts.enableConstantCodeGen = true;
    codegenOpts.enableUnrollOptimization = true;
    codegenOpts.skId = skId;
    
    ConstantCodeGenerator generator(codegenOpts);
    
    // ========== 6. 生成源码 ==========
    SK_LOGI("[ConstantCodegen] Generating constant source code...");
    SkHeaderInfo header;
    ConstantCodeGenResult genResult = generator.Generate(aicTask, aivTask, header);
    if (genResult.combinedSource.empty()) {
        SK_LOGE("[ConstantCodegen] Failed to generate constant source code");
        return {nullptr, SkKernelType::DEFAULT};
    }
    SK_LOGI("[ConstantCodegen] Source code generated, size=%zu bytes", genResult.combinedSource.size());
    
    // ========== 7. JIT 编译 ==========
    SK_LOGI("[ConstantCodegen] Starting JIT compilation...");
    ConstantCodeGenResult compileResult = generator.CompileAndResolve(genResult.combinedSource, kernelType);
    if (compileResult.specializedFuncHandle == nullptr) {
        SK_LOGE("[ConstantCodegen] JIT compilation failed");
        // 即使编译失败，也保存源码供调试
        DumpConstantCodegenFiles(skId, genResult.combinedSource, {}, kernelType, modelRI);
        return {nullptr, SkKernelType::DEFAULT};
    }
    SK_LOGI("[ConstantCodegen] JIT compilation succeeded, funcHandle=%p, binHandle=%p",
            compileResult.specializedFuncHandle, compileResult.specializedBinHandle);
    
    // ========== 8. 文件落盘 ==========
    DumpConstantCodegenFiles(skId, genResult.combinedSource, compileResult.compiledBinary, kernelType, modelRI);
    
    // ========== 9. 注册到管理器 ==========
    ConstantFuncHandleManager::Instance().RegisterFuncHandle(
        skId, compileResult.specializedFuncHandle, compileResult.specializedBinHandle);
    
    SK_LOGI("[ConstantCodegen] Constant codegen SUCCESS");
    SK_LOGI("[ConstantCodegen] Result: funcHandle=%p, kernelType=%s", 
            compileResult.specializedFuncHandle, to_string(kernelType));
    
    return {compileResult.specializedFuncHandle, kernelType};
}