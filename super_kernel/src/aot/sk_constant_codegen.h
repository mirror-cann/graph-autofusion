/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_CONSTANT_CODEGEN_H__
#define __SK_CONSTANT_CODEGEN_H__

#include "sk_common.h"
#include "sk_types.h"

#include <acl/acl.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <unordered_map>


/**
 * @brief 常量化代码生成器配置选项
 */
struct ConstantCodeGenOptions {
    bool enableConstantCodeGen = true;   // 是否启用常量化代码生成
    bool enableUnrollOptimization = true; // 是否启用 unroll 优化
    bool enableDebugEntry = false;        // 是否生成调试版本入口
    std::string outputDir;                // 输出目录
    std::string skId;                     // Super Kernel 唯一标识符
};

/**
 * @brief 生成的常量化代码结果
 */
struct ConstantCodeGenResult {
    std::string headerContent;            // 头文件内容
    std::string entrySourceContent;       // 入口函数源码内容
    std::string resolverContent;          // 函数解析器内容
    std::string combinedSource;           // 合并后的完整源码（用于 JIT 编译）
    
    // 编译后的二进制
    std::vector<uint8_t> compiledBinary;
    
    // 生成的 funcHandle（编译后解析）
    aclrtFuncHandle specializedFuncHandle = nullptr;
    aclrtBinHandle specializedBinHandle = nullptr;
};

/**
 * @brief 常量化代码生成器
 * 
 * 将运行时 TaskQue 转换为编译时常量数组，生成特化的入口函数。
 * 实现原理：
 * 1. 从 SkTask 中提取 TaskQue 信息
 * 2. 生成编译时常量结构体代码
 * 3. 生成特化的入口函数（消除 if-else 分支）
 * 4. 通过 aclrtc JIT 编译生成二进制
 * 5. 解析二进制获取 funcHandle，替换原有运行时派发的 funcHandle
 */
class ConstantCodeGenerator {
public:
    ConstantCodeGenerator(const ConstantCodeGenOptions& opts) : options_(opts) {}
    ~ConstantCodeGenerator() = default;

    /**
     * @brief 从 SkTask 生成常量化代码
     * @param aicTask AIC 任务队列
     * @param aivTask AIV 任务队列
     * @param header SkHeaderInfo 信息
     * @return 生成结果
     */
    ConstantCodeGenResult Generate(
        const SkTask& aicTask,
        const SkTask& aivTask,
        const SkHeaderInfo& header);

    /**
     * @brief 生成合并的源码（用于 aclrtc 编译）
     * @param aicTask AIC 任务队列
     * @param aivTask AIV 任务队列
     * @param kernelType 内核类型
     * @return 合并后的源码字符串
     */
    std::string GenerateCombinedSource(
        const SkTask& aicTask,
        const SkTask& aivTask,
        SkKernelType kernelType);

    /**
     * @brief 通过 aclrtc 编译源码并获取 funcHandle
     * @param source 源码字符串
     * @param kernelType 内核类型
     * @return 编译结果
     */
    ConstantCodeGenResult CompileAndResolve(
        const std::string& source,
        SkKernelType kernelType);

    /**
     * @brief 判断是否启用常量化
     */
    bool IsEnabled() const { return options_.enableConstantCodeGen; }

private:
    // 生成常量 TaskQue 代码
    std::string GenerateConstantTaskQue(
        const SkTask& task,
        const std::string& queueName);
    
    // 生成特化入口函数
    std::string GenerateSpecializedEntry(
        const SkTask& aicTask,
        const SkTask& aivTask,
        SkKernelType kernelType);
    
    // 生成针对特定 split 的任务执行代码（消除 get_coreid 运算）
    // splitIdx: split 索引 (0-3)
    std::string GenerateTaskExecutionForSplit(
        const TaskQue* taskQue,
        size_t taskIdx,
        bool isAic,
        SkKernelType preKernelType,
        int splitIdx);
    
    // 获取内核类型的模板参数
    std::pair<int, int> GetKernelTypeParams(SkKernelType kernelType);

    ConstantCodeGenOptions options_;
};

/**
 * @brief 常量化 funcHandle 管理器
 * 
 * 管理常量化生成的 funcHandle，提供运行时替换机制。
 * 线程安全实现。
 */
class ConstantFuncHandleManager {
public:
    static ConstantFuncHandleManager& Instance();

    /**
     * @brief 注册常量化生成的 funcHandle
     * @param skId Super Kernel ID
     * @param funcHandle 函数句柄
     * @param binHandle 二进制句柄
     */
    void RegisterFuncHandle(
        const std::string& skId,
        aclrtFuncHandle funcHandle,
        aclrtBinHandle binHandle);

    /**
     * @brief 获取常量化 funcHandle
     * @param skId Super Kernel ID
     * @return 函数句柄，未找到返回 nullptr
     */
    aclrtFuncHandle GetFuncHandle(const std::string& skId) const;

    /**
     * @brief 获取常量化 binHandle
     * @param skId Super Kernel ID
     * @return 二进制句柄，未找到返回 nullptr
     */
    aclrtBinHandle GetBinHandle(const std::string& skId) const;

    /**
     * @brief 检查是否存在常量化 funcHandle
     */
    bool HasFuncHandle(const std::string& skId) const;

    /**
     * @brief 清除所有缓存的 funcHandle
     */
    void Clear();

private:
    ConstantFuncHandleManager() = default;
    ~ConstantFuncHandleManager() = default;
    ConstantFuncHandleManager(const ConstantFuncHandleManager&) = delete;
    ConstantFuncHandleManager& operator=(const ConstantFuncHandleManager&) = delete;

    struct HandlePair {
        aclrtFuncHandle funcHandle = nullptr;
        aclrtBinHandle binHandle = nullptr;
    };

    std::unordered_map<std::string, HandlePair> handleMap_;
    mutable std::mutex mutex_;
};

// ==================== 全局接口函数（非命名空间） ====================

class SuperKernelOptionsManager;

// Forward declaration
typedef void* aclmdlRI;

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
    aclmdlRI modelRI);

#endif // __SK_CONSTANT_CODEGEN_H__
