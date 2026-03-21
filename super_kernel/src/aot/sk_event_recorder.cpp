/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "sk_event_recorder.h"
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/time.h>
#include <filesystem>
#include "acl/acl.h"
#include "sk_log.h"
#include "aprof_pub.h"
#include "sk_resource_manager.h"
#include "sk_file_guard.h"

SkEventRecorder& SkEventRecorder::Instance() {
    static SkEventRecorder instance;
    return instance;
}

SkEventRecorder::~SkEventRecorder() {
    SkProfilingShutdown();
}

bool SkEventRecorder::Init() {
    // 已经初始化过，直接返回
    if (enabled) {
        return true;
    }
    SK_LOGI("[sk time profiling] ===================== Start dump the time of superkernel =======================\n");
    // 检查打点环境变量是否开启（只有等于 "1" 时才启用）
    const char* env = std::getenv(ENV_SK_EVENT_RECORD);
    bool shouldEnable = (env != nullptr && std::string(env) == "1");
    if (!shouldEnable) {
        enabled = false;
        return false;
    }

    enabled = true;
    globalRunning.store(true);
    // 初始化输出目录
    outputDir = GetBasePath();
    if (outputDir.empty()) {
        SK_LOGE("[sk time profiling] Failed to get output directory, stop dumping the time information of sk\n");
        SK_LOGE("[sk time profiling] ===================== End dump the time of superkernel =======================\n");
        enabled = false;
        globalRunning.store(false);
        return false;
    }
    
    // 启动单个全局后台线程用于搬运解析记录事件
    int ret = pthread_create(&dumpThread, nullptr, DumpThreadFunc, this);
    if (ret != 0) {
        SK_LOGE("[sk time profiling] Failed to create dump thread, ret=%d\n", ret);
        enabled = false;
        return false;
    }
    
    SK_LOGI("[sk time profiling] Event recorder enabled with single dump thread\n");
    return true;
}

void* SkEventRecorder::GetGmAddrForDevice(uint32_t deviceId) {
    SK_LOGI("[sk time profiling] Start create device gm addr for device %u\n", deviceId);
    if (!enabled || deviceId >= SK_EVENT_MAX_DEVICE_NUM) {
        return nullptr;
    }
    
    SkEventDeviceCtx* ctx = &deviceCtxs[deviceId];
    
    // Double-check locking：确保每个 device 只初始化一次
    if (ctx->active.load()) {
        return ctx->gmAddr;
    }
    
    std::lock_guard<std::mutex> lock(mutex);
    if (ctx->active.load()) {
        return ctx->gmAddr;
    }
    
    // 创建上下文
    ctx = CreateDeviceCtx(deviceId);
    SK_LOGI("[sk time profiling] Device gm addr created for device %u\n", deviceId);
    
    return ctx ? ctx->gmAddr : nullptr;
}

SkEventDeviceCtx* SkEventRecorder::CreateDeviceCtx(uint32_t deviceId) {
    SkEventDeviceCtx* ctx = &deviceCtxs[deviceId];
    ctx->recorder = this; // 设置回调指针
    
    // 2. 分配 GM 内存
    aclError allocRet = SkResourceManager::ValueMemory(&ctx->gmAddr, SK_EVENT_TOTAL_SIZE);
    if (allocRet != ACL_SUCCESS || ctx->gmAddr == nullptr) {
        SK_LOGE("[sk time profiling] Failed to malloc GM for device %u, ret=%d\n", deviceId, allocRet);
        SkProfilingShutdown();
        return nullptr;
    }
    SK_LOGI("[sk time profiling] Malloc device gm addr, device %u\n", deviceId);
    // 3. 分配 host 缓冲区（使用智能指针，RAII 管理）
    ctx->hostBuf = std::make_unique<uint8_t[]>(SK_EVENT_TOTAL_SIZE);
    if (ctx->hostBuf == nullptr) {
        SK_LOGE("[sk time profiling] Failed to allocate host buf for device %u\n", deviceId);
        SkProfilingShutdown();
        return nullptr;
    }
    errno_t memRet = memset_s(ctx->hostBuf.get(), SK_EVENT_TOTAL_SIZE, 0, SK_EVENT_TOTAL_SIZE);
    if (memRet != EOK) {
        SK_LOGE("[sk time profiling] memset_s hostBuf failed for device %u, ret=%d\n", deviceId, memRet);
        SkProfilingShutdown();
        return nullptr;
    }
    
    // 4. 创建临时输出文件（用于存储 node 事件）
    char filename[SPRINT_LEN_BUFFER];
    errno_t snpRet = snprintf_s(filename, sizeof(filename), sizeof(filename) - 1,
                            "%s/sk_event_dev_device_%u.tmp.json", outputDir.c_str(), deviceId);
    if (snpRet < 0) {
        SK_LOGE("[sk time profiling] snprintf_s filename failed for device %u, ret=%d\n", deviceId, snpRet);
        SkProfilingShutdown();
        return nullptr;
    }
    
    // 直接使用 ctx->outputFp 打开文件
    if (ctx->outputFp.Open(filename, "w+b")) {
        // 写入 JSON 数组开头
        const char* jsonStart = "[{}]";
        size_t written = fwrite(jsonStart, 1, strlen(jsonStart), ctx->outputFp.Get());
        if (written != strlen(jsonStart)) {
            SK_LOGE("[sk time profiling] Failed to write JSON start to file\n");
            SkProfilingShutdown();
            return nullptr;
        }
    }
    
    // 5. 初始化host偏移量数组
    for (uint32_t i = 0; i < SK_EVENT_CORE_NUM; i++) {
        ctx->lastOffset[i] = sizeof(SkKernelEventCoreBuf);
    }

    // 6. 设置该device基本信息并标记为激活
    ctx->deviceId = deviceId;
    ctx->totalSize = SK_EVENT_TOTAL_SIZE;
    ctx->active.store(1);
    
    SK_LOGI("[sk time profiling] Created context for device %u, GM addr=%p\n", 
           deviceId, ctx->gmAddr);
    
    return ctx;
}

// core id 大于25就是aiv
static bool CoreIsAiv(int coreId) {
    return coreId >= 25;
}

void SkEventRecorder::DumpModelData(SkEventRecorder* recorder) {
    // 合并临时文件：先写聚合事件，再追加 node 事件, 避免chrome tracing显示的时候重叠
    for (uint32_t i = 0; i < SK_EVENT_MAX_DEVICE_NUM; i++) {
        SkEventDeviceCtx* ctx = &recorder->deviceCtxs[i];
        if (ctx->active.load() && ctx->outputFp.IsValid()) {
            SK_LOGI("[sk time profiling] Start dump model time, device:%u\n", i);
            char finalFile[SPRINT_LEN_BUFFER];
            errno_t finalRet = snprintf_s(finalFile, sizeof(finalFile), sizeof(finalFile) - 1,
                                    "%s/sk_event_dev_device_%u.json", recorder->outputDir.c_str(), i);
            if (finalRet < 0) {
                SK_LOGE("[sk time profiling] snprintf_s finalFile failed for device %u, ret=%d\n", i, finalRet);
                SkProfilingShutdown();
                return;
            }

            // 创建最终文件并写入聚合事件
            FileGuard finalFp(finalFile, "wb");
            if (finalFp.IsValid()) {
                // 写入 JSON 数组开头
                const char* jsonStart = "[{}";
                fwrite(jsonStart, 1, strlen(jsonStart), finalFp.Get());

                // 输出 model 级别事件
                if (!WriteModelEventsToJson(ctx, finalFp.Get())) {
                    SK_LOGE("[sk time profiling] Failed to write model events to JSON for device %u\n", i);
                    SkProfilingShutdown();
                    return;
                }
                
                // 输出 sk 级别事件
                if (!WriteSkEventsToJson(ctx, finalFp.Get())) {
                    SK_LOGE("[sk time profiling] Failed to write sk events to JSON for device %u\n", i);
                    SkProfilingShutdown();
                    return;
                }

                // 追加临时文件内容（跳过开头的 "["）
                fflush(ctx->outputFp.Get());
                fseek(ctx->outputFp.Get(), 0, SEEK_END);
                long tmpSize = ftell(ctx->outputFp.Get());
                fseek(ctx->outputFp.Get(), 1, SEEK_SET);  // 跳过 "["

                const char* jsonEnd = ",\n";
                fwrite(jsonEnd, 1, strlen(jsonEnd), finalFp.Get());
                
                char buf[8192];
                size_t n;
                while ((n = fread(buf, 1, sizeof(buf), ctx->outputFp.Get())) > 0) {
                    fwrite(buf, 1, n, finalFp.Get());
                }
                
                fflush(finalFp.Get());
                // finalFp 由 FileGuard 自动关闭
            }

            // 关闭并删除临时文件
            ctx->outputFp.Close();
            // 删除子算子耗时信息临时文件路径, 如果程序意外退出，则保留子算子耗时信息文件
            char tmpFile[SPRINT_LEN_BUFFER];
            errno_t tmpRet = snprintf_s(tmpFile, sizeof(tmpFile), sizeof(tmpFile) - 1,
                          "%s/sk_event_dev_device_%u.tmp.json", recorder->outputDir.c_str(), i);
            if (tmpRet < 0) {
                SK_LOGE("[sk time profiling] snprintf_s tmpFile failed for device %u, ret=%d\n", i, tmpRet);
            } else {
                remove(tmpFile);
            }
            SK_LOGI("[sk time profiling] End dump model time, device:%u\n", i);
        }
    }
}

void* SkEventRecorder::DumpThreadFunc(void* arg) {
    SkEventRecorder* recorder = static_cast<SkEventRecorder*>(arg);
    
    SK_LOGI("[sk time profiling] Global dump thread started\n");
    
    while (recorder->globalRunning.load()) {
        // 遍历所有 device，处理每个激活的 device
        for (uint32_t i = 0; i < SK_EVENT_MAX_DEVICE_NUM; i++) {
            SkEventDeviceCtx* ctx = &recorder->deviceCtxs[i];
            if (ctx->active.load()) {
                // 写入node的耗时信息
                SK_LOGI("[sk time profiling] Nodes dump thread started, device:%u\n", i);
                recorder->DumpDeviceData(ctx);
                SK_LOGI("[sk time profiling] Nodes dump thread stopped, device:%u\n", i);
            }
        }
        usleep(100000); // 100ms 轮询间隔
    }
    
    // 最后一次刷新所有 device
    for (uint32_t i = 0; i < SK_EVENT_MAX_DEVICE_NUM; i++) {
        SkEventDeviceCtx* ctx = &recorder->deviceCtxs[i];
        if (ctx->active.load()) {
            SK_LOGI("[sk time profiling] Nodes dump thread started, device:%u\n", i);
            recorder->DumpDeviceData(ctx);
            SK_LOGI("[sk time profiling] Nodes dump thread stopped, device:%u\n", i);
        }
    }
    
    // 往json补充model和sk信息
    recorder->DumpModelData(recorder);
    
    // 复制 JSON 文件到 mindstudio_profiler_output路径
    if (recorder->outputDir.empty()) {
        SK_LOGE("[sk time profiling] mindstudio_profiler_output floder is not find, sk profiling file will store in PROF floder\n");
        recorder->SkProfilingShutdown();
        return nullptr;
    } else {
        for (uint32_t i = 0; i < SK_EVENT_MAX_DEVICE_NUM; i++) {
            SkEventDeviceCtx* ctx = &recorder->deviceCtxs[i];
            if (ctx->active.load()) {
                std::string srcFile = recorder->outputDir + "/sk_event_dev_device_" + std::to_string(i) + ".json";
                std::string dstFile = recorder->outputDir + "/mindstudio_profiler_output/sk_event_dev_device_" + std::to_string(i) + ".json";

                // 复制文件
                std::error_code ec;
                if (!std::filesystem::copy_file(srcFile, dstFile, std::filesystem::copy_options::overwrite_existing, ec)) {
                    SK_LOGE("[sk time profiling] Failed to copy file from %s to %s: %s\n", 
                            srcFile.c_str(), dstFile.c_str(), ec.message().c_str());
                    recorder->SkProfilingShutdown();
                    return nullptr;
                }
                // 删除算子耗时的临时文件
                remove(srcFile.c_str());
                SK_LOGI("[sk time profiling] Successfully saved profiling file to mindstudio_profiler_output: %s\n", dstFile.c_str());
            }
        }
    }

    SK_LOGI("[sk time profiling] Global dump thread stopped\n");
    return nullptr;
}

bool SkEventRecorder::WriteNodeEventToJson(SkEventDeviceCtx* ctx, const SkKernelEventRecord* record,
                                            uint32_t core, const SkNodeInfo& nodeInfo) {
    if (!ctx->outputFp.IsValid()) {
        SK_LOGE("[sk time profiling] Failed to open the output file\n");
        return false;  // 文件未打开，跳过写入
    }
    
    // 移动到 "}]" 之前
    if (fseek(ctx->outputFp.Get(), -2, SEEK_END) != 0) {
        SK_LOGE("[sk time profiling] Failed to seek in output file\n");
        ctx->outputFp.Close();
        SkProfilingShutdown();
        return false;
    }
    
    double tsStart = (double)record->startTime / TICK_US_MULTIPLER;
    double tsEnd = (double)record->endTime / TICK_US_MULTIPLER;
    char jsonLine[SPRINT_LEN_BUFFER];
    int len = snprintf_s(jsonLine, sizeof(jsonLine), sizeof(jsonLine) - 1,
        "\"ph\":\"X\",\"name\":\"[%u/%u]%s\",\"pid\":\"%s\",\"tid\":%u,"
        "\"ts\":%f,\"dur\":%f,\"args\":{\"modelRI\":%lu,\"skId\":%u,\"nodeId\":%u,\"nodeName\":\"%s\",\"deviceId\":%u,\"coreId\":%u}},\n{}]",
        record->blockIdx, record->blockNum, nodeInfo.nodeName.c_str(), CoreIsAiv(core) ? "AIV" : "AIC", core,
        tsStart, (record->endTime > record->startTime) ? (tsEnd - tsStart) : 0,
        record->modelRI, record->skId, record->nodeId,
        nodeInfo.nodeName.c_str(), ctx->deviceId, core);
    
    if (len < 0) {
        SK_LOGE("[sk time profiling] snprintf_s failed for JSON line, modelRI=%lu, skId=%u, nodeId=%u\n", 
                record->modelRI, record->skId, record->nodeId);
        ctx->outputFp.Close();
        SkProfilingShutdown();
        return false;
    } else if (len > 0 && len < sizeof(jsonLine)) {
        size_t written = fwrite(jsonLine, 1, len, ctx->outputFp.Get());
        if (written != static_cast<size_t>(len)) {
            SK_LOGE("[sk time profiling] Failed to write JSON line\n");
            ctx->outputFp.Close();
            SkProfilingShutdown();
            return false;
        }
    } else {
        SK_LOGW("[sk time profiling] JSON line too long or truncated, modelRI=%lu, skId=%u, nodeName=%s,len=%d\n", 
                record->modelRI, record->skId, nodeInfo.nodeName.c_str(), len);
    }
    return true;
}

void SkEventRecorder::UpdateTimeStats(SkEventDeviceCtx* ctx, const SkKernelEventRecord* record, uint32_t core) {
    // 更新 sk 级别统计信息：modelRI -> skId -> coreId
    SkCoreTimeStats& skStats = ctx->skCoreTimeStats[record->modelRI][record->skId][core];
    if (record->startTime < skStats.minStartTime) {
        skStats.minStartTime = record->startTime;
        skStats.blockIdx = record->blockIdx;
        skStats.blockNum = record->blockNum;
    }
    if (record->endTime > skStats.maxEndTime) {
        skStats.maxEndTime = record->endTime;
    }

    // 更新 model 级别统计信息：modelRI -> coreId
    SkCoreTimeStats& modelStats = ctx->modelCoreTimeStats[record->modelRI][core];
    if (record->startTime < modelStats.minStartTime) {
        modelStats.minStartTime = record->startTime;
    }
    if (record->endTime > modelStats.maxEndTime) {
        modelStats.maxEndTime = record->endTime;
    }
}

bool SkEventRecorder::WriteModelEventsToJson(SkEventDeviceCtx* ctx, FILE* finalFp) {
    // 输出 model 级别事件：modelRI -> coreId -> stats：先输出 AIC，再输出 AIV
    for (int pass = 0; pass < 2; pass++) {
        for (auto& modelIt : ctx->modelCoreTimeStats) {
            uint64_t modelRI = modelIt.first;
            for (auto& coreIt : modelIt.second) {
                uint32_t core = coreIt.first;
                SkCoreTimeStats& stats = coreIt.second;
                
                // pass 0: 只输出 AIC（非 AIV）
                // pass 1: 只输出 AIV
                bool isAiv = CoreIsAiv(core);
                if ((pass == 0 && isAiv) || (pass == 1 && !isAiv)) {
                    continue;
                }
                
                if (stats.minStartTime < UINT64_MAX && stats.maxEndTime > 0) {
                    double tsStart = (double)stats.minStartTime / TICK_US_MULTIPLER;
                    double tsEnd = (double)stats.maxEndTime / TICK_US_MULTIPLER;
                    
                    char jsonLine[SPRINT_LEN_BUFFER];
                    int len = snprintf_s(jsonLine, sizeof(jsonLine), sizeof(jsonLine) - 1,
                        ",\n{\"ph\":\"X\",\"name\":\"modelRI: %lu\",\"pid\":\"%s\",\"tid\":%u,"
                        "\"ts\":%f,\"dur\":%f,\"args\":{\"deviceId\":%u,\"coreId\":%u}}",
                        modelRI, isAiv ? "AIV" : "AIC", core,
                        tsStart, (tsEnd > tsStart) ? (tsEnd - tsStart) : 0,
                        ctx->deviceId, core);
                    
                    if (len < 0) {
                        SK_LOGE("[sk time profiling] snprintf_s failed for model event JSON line, modelRI=%lu, core=%u\n", 
                                modelRI, core);
                        return false;
                    } else if (len > 0) {
                        size_t written = fwrite(jsonLine, 1, len, finalFp);
                        if (written != static_cast<size_t>(len)) {
                            SK_LOGE("[sk time profiling] Failed to write model event JSON line\n");
                            return false;
                        }
                        SK_LOGI("[sk time profiling] Add model time event for device %u, core %u\n", 
                            ctx->deviceId, core);
                    }
                }
            }
        }
    }
    return true;
}

bool SkEventRecorder::WriteSkEventsToJson(SkEventDeviceCtx* ctx, FILE* finalFp) {
    // 输出 sk 级别事件：modelRI -> skId -> coreId -> stats
    for (auto& modelIt : ctx->skCoreTimeStats) {
        uint64_t modelRI = modelIt.first;
        for (auto& skIt : modelIt.second) {
            uint32_t skId = skIt.first;
            for (auto& coreIt : skIt.second) {
                uint32_t core = coreIt.first;
                SkCoreTimeStats& stats = coreIt.second;
                
                if (stats.minStartTime < UINT64_MAX && stats.maxEndTime > 0) {
                    double tsStart = (double)stats.minStartTime / TICK_US_MULTIPLER;
                    double tsEnd = (double)stats.maxEndTime / TICK_US_MULTIPLER;
                    
                    char jsonLine[SPRINT_LEN_BUFFER];
                    int len = snprintf_s(jsonLine, sizeof(jsonLine), sizeof(jsonLine) - 1,
                        ",\n{\"ph\":\"X\",\"name\":\"[%u/%u]skId: %u\",\"pid\":\"%s\",\"tid\":%u,"
                        "\"ts\":%f,\"dur\":%f,\"args\":{\"deviceId\":%u,\"coreId\":%u}}",
                        stats.blockIdx, stats.blockNum, skId, CoreIsAiv(core) ? "AIV" : "AIC", core,
                        tsStart, (tsEnd > tsStart) ? (tsEnd - tsStart) : 0,
                        ctx->deviceId, core);
                    
                    if (len < 0) {
                        SK_LOGE("[sk time profiling] snprintf_s failed for sk event JSON line, skId=%u, core=%u\n", 
                                skId, core);
                        return false;
                    } else if (len > 0) {
                        size_t written = fwrite(jsonLine, 1, len, finalFp);
                        if (written != static_cast<size_t>(len)) {
                            SK_LOGE("[sk time profiling] Failed to write sk event JSON line\n");
                            return false;
                        }
                        SK_LOGI("[sk time profiling] Add sk time event for device %u, core %u\n", 
                            ctx->deviceId, core);
                    }
                }
            }
        }
    }
    return true;
}

void SkEventRecorder::DumpDeviceData(SkEventDeviceCtx* ctx) {
    if (ctx->gmAddr == nullptr || ctx->hostBuf == nullptr) {
        return;
    }

    // 拷贝 GM 到 host
    aclError ret = aclrtMemcpy(ctx->hostBuf.get(), ctx->totalSize, 
                ctx->gmAddr, ctx->totalSize, 
                ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("[sk time profiling] Failed to memcpy from device %u, ret=%d\n", ctx->deviceId, ret);
        SkProfilingShutdown();
        return;
    }
    
    uint8_t* hostBuf = ctx->hostBuf.get();
    bool hasNewData = false;
    
    // 遍历每个 core 的数据
    for (uint32_t core = 0; core < SK_EVENT_CORE_NUM; core++) {
        SkKernelEventCoreBuf* coreBuf = reinterpret_cast<SkKernelEventCoreBuf*>(
            hostBuf + core * SK_EVENT_CORE_SIZE);
        
        uint32_t curOffset = coreBuf->offset;
        uint64_t lastOff = ctx->lastOffset[core];
        
        // 读取新数据
        while (lastOff + sizeof(SkKernelEventRecord) <= curOffset && 
               lastOff + sizeof(SkKernelEventRecord) <= SK_EVENT_CORE_SIZE) {
            SkKernelEventRecord* record = reinterpret_cast<SkKernelEventRecord*>(
                hostBuf + core * SK_EVENT_CORE_SIZE + lastOff);

            // 查询 NodeInfo
            SkNodeInfo nodeInfo = SkEventRecorder::Instance().GetNodeInfo(record->modelRI, record->skId, record->nodeId);
            if (nodeInfo.nodeName != "") {
                // 写入 JSON trace 文件
                if (!WriteNodeEventToJson(ctx, record, core, nodeInfo)) {
                    SK_LOGE("[sk time profiling] Failed to write node event to json, device %u, ret=%d\n", ctx->deviceId, ret);
                    return;
                }
                
                // 更新model和sk大算子的统计信息
                UpdateTimeStats(ctx, record, core);
            }
            
            hasNewData = true;
            lastOff += sizeof(SkKernelEventRecord);
        }
        
        ctx->lastOffset[core] = lastOff;
        if (lastOff + sizeof(SkKernelEventRecord) > SK_EVENT_CORE_SIZE) {
            // 缓冲区已满，后续不再打印
            SK_LOGW("[sk time profiling]  device %u core %u buffer is full, stop dump the time of nodes on device %u core %u\n", ctx->deviceId, core, ctx->deviceId, core);
        }
    }
    
    if (hasNewData && ctx->outputFp.IsValid()) {
        fflush(ctx->outputFp.Get());
    }
    SK_LOGI("[sk time profiling] Add some node event, device %u\n", ctx->deviceId);
}

void SkEventRecorder::SkProfilingShutdown() {
    if (!enabled) {
        return;
    }
    
    SK_LOGI("[sk time profiling] Shutting down event recorder\n");
    
    // 停止全局后台线程
    globalRunning.store(false);
    pthread_join(dumpThread, nullptr);
    
    // 释放所有资源 device空间由SkResourceManager独立负责释放
    for (uint32_t i = 0; i < SK_EVENT_MAX_DEVICE_NUM; i++) {
        SkEventDeviceCtx* ctx = &deviceCtxs[i];
        if (ctx->active.load()) {
            // hostBuf 由 unique_ptr 自动释放，无需手动 free
            ctx->hostBuf.reset();
            ctx->active.store(0);
        }
    }
    enabled = false;
}

void SkEventRecorder::AddNodeInfoMapping(uint64_t modelRI, uint32_t skId, uint32_t nodeId,
                                          const std::string& nodeName, uint32_t numBlocks) {
    std::lock_guard<std::mutex> lock(nodeInfoMapMutex);
    SkNodeInfo info;
    info.nodeName = nodeName;
    info.numBlocks = numBlocks;
    nodeInfoMap[modelRI][skId][nodeId] = info;
}

SkNodeInfo SkEventRecorder::GetNodeInfo(uint64_t modelRI, uint32_t skId, uint32_t nodeId) const {
    std::lock_guard<std::mutex> lock(nodeInfoMapMutex);
    SkNodeInfo emptyInfo;
    
    auto modelIt = nodeInfoMap.find(modelRI);
    if (modelIt == nodeInfoMap.end()) {
        return emptyInfo;
    }
    auto skIt = modelIt->second.find(skId);
    if (skIt == modelIt->second.end()) {
        return emptyInfo;
    }
    auto nodeIt = skIt->second.find(nodeId);
    if (nodeIt == skIt->second.end()) {
        return emptyInfo;
    }
    return nodeIt->second;
}

// ==================== 性能分析相关函数 ====================

const char* GetEntryFuncNameByOpType(SkKernelType& opType) {
    // sk_entry_aiv
    if (opType == SkKernelType::AIV_ONLY || opType == SkKernelType::MIX_AIV_1_0) {
        return "sk_entry_aiv";
    }

    // sk_entry_aic
    if (opType == SkKernelType::AIC_ONLY || opType == SkKernelType::MIX_AIC_1_0) {
        return "sk_entry_aic";
    }
    // sk_entry_mix11
    if (opType == SkKernelType::MIX_AIC_1_1) {
        return "sk_entry_mix11";
    }
    // sk_entry_mix12
    if (opType == SkKernelType::MIX_AIC_1_2) {
        return "sk_entry_mix12";
    }

    // Unknown opType
    SK_LOGE("opType is not in the enum class SkKernelType");
    return nullptr;
}

bool SkProfiling(const SuperKernelProcessedScopeInfo &scopeInfo, SkLaunchInfo &launchInfo,
                                        SuperKernelGraph& graph) {
    SK_LOGI("[sk shape profiling] =============== Start shape profiling ===================");
    SkHostEntryInfo& skEntryInfo = launchInfo.entryInfo;
    
    uint32_t taskType = MSPROF_GE_TASK_TYPE_MIX_AIC; //全部填mix_aic 4
    uint32_t opFlag = 0; //记录op属性标记的bitmap，bit0代表是否使能了HF32
    std::string combinedAttrIdStr;
    uint32_t maxTensorNum = SHAPE_MAX_TENSOR_NUM;

    // ====== 第一遍遍历：计算总 tensor 数量，并收集 NODE_KERNEL 类型的节点 ======
    uint32_t totalTensorNum = 0;
    std::vector<SuperKernelBaseNode*> kernelNodes;
    for (size_t i = 0; i < scopeInfo.nodes.size(); ++i) {
        SuperKernelBaseNode* node = scopeInfo.nodes[i];
        if (node == nullptr) {
            SK_LOGE("[sk profiling] Failed to get node, node is nullptr");
            return false;
        }
        SkNodeType nodeType = node->GetNodeType();
        if (nodeType == SkNodeType::NODE_KERNEL) {
            const NodeInfos& nodeInfos = node->GetNodeInfos();
            const KernelInfos& kernelInfos = nodeInfos.kernelInfos;
            if (kernelInfos.opInfoPtr != nullptr) {
                const CacheopInfoBasic* cacheInfo = 
                    static_cast<const CacheopInfoBasic*>(kernelInfos.opInfoPtr);
                totalTensorNum += cacheInfo->tensorNum;
            }

            if (totalTensorNum > maxTensorNum) {
                totalTensorNum = maxTensorNum; // 截断到最大值
                break;
            }
            kernelNodes.push_back(node);
        }
    }

    size_t totalSize = sizeof(CacheopInfoBasic) + totalTensorNum * sizeof(MsrofTensorData);
    auto ptr = std::make_unique<uint8_t[]>(totalSize);
    if (ptr == nullptr) {
        SK_LOGE("[sk profiling] Failed to allocate memory for launchInfo.cacheInfo\n");
        return false;
    }

    // ptr生命周期跟随graph，因为runtime在aclmdlRIUpdate才会把shape信息copy走
    launchInfo.cacheInfo = ptr.get();
    graph.AddShapeInfoPtr(std::move(ptr));
    errno_t ret = memset_s(launchInfo.cacheInfo, totalSize, 0, totalSize);
    if (ret != EOK) {
        SK_LOGE("[sk profiling] memset_s launchInfo.cacheInfo failed, ret=%d\n",  ret);
    }
    launchInfo.cacheopInfoSize = totalSize;
    uint8_t *dest = static_cast<uint8_t *>(launchInfo.cacheInfo);
    uint64_t destOffset = sizeof(CacheopInfoBasic); // 预留 CacheopInfoBasic 的空间，后续填充

    // ====== 第二遍遍历：从 kernelNodes 收集数据 ======
    for (size_t i = 0; i < kernelNodes.size(); ++i) {
        SuperKernelBaseNode* node = kernelNodes[i];
        
        const NodeInfos& nodeInfos = node->GetNodeInfos();
    
        const KernelInfos& kernelInfos = nodeInfos.kernelInfos;
        if (kernelInfos.opInfoPtr != nullptr) {
            const CacheopInfoBasic* cacheInfo = 
                static_cast<const CacheopInfoBasic*>(kernelInfos.opInfoPtr);
            if (kernelInfos.opInfoSize >= (sizeof(CacheopInfoBasic) + sizeof(MsrofTensorData) * cacheInfo->tensorNum)) {
                // attrId
                char* attrIdStr = MsprofId2Str(cacheInfo->attrId);
                if (!combinedAttrIdStr.empty()) {
                    combinedAttrIdStr += "|";
                }
                combinedAttrIdStr += attrIdStr;
                
                // opFlag 取值只有0和1
                opFlag = opFlag || cacheInfo->opFlag;
                
                // ====== 复制 tensorData 到 launchInfo.cacheInfo ======
                for (uint32_t t = 0; t < cacheInfo->tensorNum; ++t) {
                    const MsrofTensorData& tensor = cacheInfo->tensorData[t];
                    MsrofTensorData msTensor;
                    msTensor.tensorType = tensor.tensorType;
                    msTensor.format = tensor.format;
                    msTensor.dataType = tensor.dataType;
                    for (int s = 0; s < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++s) {
                        msTensor.shape[s] = tensor.shape[s];
                    }
                    errno_t ret = memcpy_s(dest + destOffset, totalSize - destOffset,
                                            &msTensor, sizeof(MsrofTensorData));
                    if (ret != EOK) {
                        SK_LOGE("[sk profiling] memcpy_s failed, ret=%d\n",  ret);
                        return false;
                    }
                    destOffset += sizeof(MsrofTensorData);
                }
            }
            else {
                SK_LOGE("[sk profiling] warning: kernelInfos.opInfoSize should be greater than or equal to kernelInfos.opInfoPtr \n");
            }
        }
    }

    // 将拼接后的字符串转换回 uint64_t
    uint64_t combinedAttrId = combinedAttrIdStr.empty() ? 0 : 
        MsprofStr2Id(combinedAttrIdStr.c_str(), combinedAttrIdStr.length());

    CacheopInfoBasic cacheopInfoBasic;
    cacheopInfoBasic.taskType = taskType;
    // numBlocks 编码：高16位表示mix模式类型，低16位为实际numBlocks
    // sk_entry_mix11 -> 高16位=1, sk_entry_mix12 -> 高16位=2
    uint32_t numBlocks = skEntryInfo.numBlocks;
    if (skEntryInfo.entryType == SkKernelType::MIX_AIC_1_1) {
        cacheopInfoBasic.numBlocks = (1U << 16) | (numBlocks & 0xFFFF);
    } else if (skEntryInfo.entryType == SkKernelType::MIX_AIC_1_2) {
        cacheopInfoBasic.numBlocks = (2U << 16) | (numBlocks & 0xFFFF);
    } else {
        cacheopInfoBasic.numBlocks = numBlocks;
    }
    const char* skEntryFuncName = GetEntryFuncNameByOpType(skEntryInfo.entryType);
    if (skEntryFuncName != nullptr) {
        cacheopInfoBasic.nodeId = MsprofStr2Id(skEntryFuncName, strlen(skEntryFuncName));
    } else {
        SK_LOGE("[sk profiling] Failed to get entry func name\n");
        return false;
    }
    const char* opTypeStr = "SuperKernel";
    cacheopInfoBasic.opType = MsprofStr2Id(opTypeStr, strlen(opTypeStr));
    cacheopInfoBasic.attrId = combinedAttrId;
    cacheopInfoBasic.opFlag = opFlag;
    cacheopInfoBasic.tensorNum = totalTensorNum; 
    ret = memcpy_s(dest, sizeof(CacheopInfoBasic), &cacheopInfoBasic, sizeof(CacheopInfoBasic));
    if (ret != EOK) {
        SK_LOGE("[sk profiling] memcpy_s cacheopInfoBasic failed, ret=%d\n",  ret);
    }
    const CacheopInfoBasic* cacheInfoPtr = static_cast<const CacheopInfoBasic*>(launchInfo.cacheInfo);
    SK_LOGI("[sk shape profiling] sk shape information verify: taskType=%u, numBlocks=%u, nodeId=%lu, opType=%lu, attrId=%lu, opFlag=%u, tensorNum=%u, infoSize=%lu\n",
            cacheInfoPtr->taskType, cacheInfoPtr->numBlocks, cacheInfoPtr->nodeId,
            cacheInfoPtr->opType, cacheInfoPtr->attrId, cacheInfoPtr->opFlag, cacheInfoPtr->tensorNum, launchInfo.cacheopInfoSize);
    SK_LOGI("[sk shape profiling] =============== End shape profiling ===================");
    return true;
}

bool DumpProfilingDetail(const std::vector<SuperKernelBaseNode *> &taskNodes, SkLaunchInfo &launchInfo,
                                const SuperKernelProcessedScopeInfo &scopeInfo, aclmdlRI modelRI) {
    // 获取事件记录 GM 地址并更新 devArgs 中的事件配置
    if (SkEventRecorder::Instance().IsEnabled()) {
        int32_t deviceId = 0;
        aclrtGetDevice(&deviceId);
        launchInfo.eventGmAddr = SkEventRecorder::Instance().GetGmAddrForDevice(deviceId);
        if (launchInfo.eventGmAddr == nullptr) {
                SK_LOGE("[sk time profiling] Failed to get event GM address\n");
                return false;
            }
        launchInfo.modelRI = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(modelRI));  // modelRI只有一个void*，先用hash值作为modelRI的id，后续可以改成更合理的来源
        launchInfo.skId = scopeInfo.scopeIdx;  
        SK_LOGI("[sk time profiling] Event recording enabled, gm_addr=%p, modelRI=%lu, skId=%u\n", launchInfo.eventGmAddr, launchInfo.modelRI, launchInfo.skId);
        
        // 更新 devArgs 中的事件配置
        if (launchInfo.devArgs.Get() != nullptr && 
            launchInfo.devArgs.Get()->skHeader.eventConfigOffset != 0) {
            uint8_t* base = reinterpret_cast<uint8_t*>(launchInfo.devArgs.Get());
            SkEventConfig* eventConfig = reinterpret_cast<SkEventConfig*>(
                base + launchInfo.devArgs.Get()->skHeader.eventConfigOffset);
            eventConfig->eventGmAddr = reinterpret_cast<uint64_t>(launchInfo.eventGmAddr);
            eventConfig->modelRI = launchInfo.modelRI;
            eventConfig->skId = launchInfo.skId;
            eventConfig->enabled = 1;
        }
        
        // 建立 modelRI -> skId -> nodeId -> (nodeName, numBlocks) 映射
        for (size_t nodeId = 0; nodeId < taskNodes.size(); ++nodeId) {
            SuperKernelBaseNode* node = taskNodes[nodeId];
            if (node != nullptr && node->GetNodeType() == SkNodeType::NODE_KERNEL) {
                const NodeInfos& nodeInfos = node->GetNodeInfos();
                const std::string& funcName = nodeInfos.kernelInfos.funcName;
                // launchInfo.entryInfo.numBlocks是sk大算子的numBlocks
                SkEventRecorder::Instance().AddNodeInfoMapping(
                    launchInfo.modelRI, launchInfo.skId, static_cast<uint32_t>(nodeId), funcName, nodeInfos.kernelInfos.numBlocks);
            }
        }
    } else {
        launchInfo.eventGmAddr = nullptr;
        launchInfo.modelRI = 0;
        launchInfo.skId = 0;
    }
    return true;
}