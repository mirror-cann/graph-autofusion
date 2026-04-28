/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "super_kernel.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_optimizer.h"
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_dump_json.h"
#include "sk_dfx_exception_handler.h"
#include "sk_lock_detector.h"
#include "sk_resource_manager.h"
#include "sk_scope_launch.h"
#include "sk_event_recorder.h"

namespace {
class CurrentModelGuard {
public:
    explicit CurrentModelGuard(aclmdlRI model)
    {
        SkResourceManager::SetCurrentModel(model);
    }

    ~CurrentModelGuard()
    {
        SkResourceManager::SetCurrentModel(nullptr);
    }
};

/**
 * @brief Dump graph to JSON file for debugging
 * @param model Model RI handle
 * @param metaDir Meta directory path
 * @param deviceId Device ID
 * @param suffix Filename suffix (e.g., "before" or "after")
 * @return aclError status
 */
aclError DumpGraphJson(aclmdlRI model, const std::string& metaDir, int32_t deviceId, const std::string& suffix) {
    if (!sk::logger::FileLogger::Instance().IsEnabled()) {
        return ACL_SUCCESS;  // Kernel meta save is disabled, skip
    }
    std::string jsonPath = metaDir + "/" + suffix + "_" + std::to_string(deviceId) + ".json";
    aclError ret = aclmdlRIDebugJsonPrint(model, jsonPath.c_str(), 0);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to print json file: %s", jsonPath.c_str());
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

/**
 * @brief Get device ID and create meta directory for graph dumping
 * @param model Model RI handle
 * @param deviceId Output device ID
 * @param metaDir Output meta directory path
 * @return aclError status
 */
aclError PrepareGraphDumpEnv(aclmdlRI model, int32_t& deviceId, std::string& metaDir) {
    if (!sk::logger::FileLogger::Instance().IsEnabled()) {
        return ACL_SUCCESS;  // Kernel meta save is disabled, skip directory creation
    }
    aclError ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to get device id.");
        return ACL_ERROR_FAILURE;
    }
    metaDir = CreateSkMetaDirectory(model);
    return ACL_SUCCESS;
}
}

#ifdef __cplusplus
extern "C" {
#endif

aclError aclskOptimize(aclmdlRI model, aclskOptions *options) {

    // Initialize logger first (controlled by environment variable ASCEND_OP_COMPILE_SAVE_KERNEL_META)
    InitSkLogger(model);

    int32_t deviceId;
    std::string metaDir;
    aclError ret = PrepareGraphDumpEnv(model, deviceId, metaDir);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    // Dump raw task information from modelRI to JSON (before fusion)
    SK_LOGI("Start dump raw tasks from modelRI to JSON...");
    if (!DumpModelRITasksToJson(model, "sk_raw_tasks_before", deviceId)) {
        SK_LOGW("Failed to dump raw tasks to JSON, continuing...");
    }

    ret = DumpGraphJson(model, metaDir, deviceId, "sk_graph_rts_before");
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    CurrentModelGuard modelGuard(model);
    ret = aclrtSetExceptionInfoCallback(SuperKernelExceptionCallBackFunc);
    if (ret != ACL_SUCCESS) {
        SK_LOGE("Failed to set exception callback.");
        return ACL_ERROR_FAILURE;
    }

    SK_LOGI("Begin aclskOptimize");
    SK_LOGI("Start init sk graph...");
    SuperKernelGraph graph(model);
    if (!graph.InitSKGraph()) {
        return ACL_ERROR_FAILURE;
    }

    SK_LOGI("End init sk graph, start dump graph to JSON...");
    if (!DumpGraphNodesToJson(graph)) {
        SK_LOGW("Failed to dump graph nodes to JSON, continuing...");
    }

    ret = LockDetector::GetDeviceCores();
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    SK_LOGI("End init sk graph");
    
    SK_LOGI("Start init sk time profiling event recorder...");
    SkEventRecorder::Instance().Init(); // Initialize event recorder (if ASCEND_PROF_SK_ON=1 environment variable is set)
    SK_LOGI("End init sk time profiling event recorder");

    SK_LOGI("Start parse sk options...");
    SuperKernelOptionsManager opts;
    opts.ParseOptions(options);
    SK_LOGI("End parse sk options");

    SK_LOGI("Start optimize sk graph...");
    SuperKernelOptimizer optimizer(opts);
    if (!optimizer.Process(graph)) {
        SK_LOGE("aclskOptimize failed: optimize sk graph failed");
        return ACL_ERROR_FAILURE;
    }
    SK_LOGI("End optimize sk graph");

    SK_LOGI("Start update graph...");
    ret = graph.Update();
    SK_LOGI("End update graph");

    // Dump raw task information from modelRI to JSON (after fusion/update) with options
    SK_LOGI("Start dump raw tasks after update from modelRI to JSON...");
    if (!DumpModelRITasksToJsonWithOpts(model, "sk_raw_tasks_after", deviceId, opts)) {
        SK_LOGW("Failed to dump raw tasks to JSON, continuing...");
    }

    // Dump fused graph to JSON with nested structure (after update to capture updated values)
    SK_LOGI("Start dump fused graph to JSON...");
    if (!DumpFusedGraphToJson(graph, optimizer.GetScopeInfos())) {
        SK_LOGW("Failed to dump fused graph to JSON, continuing...");
    }

    // Dump kernel binaries to bin_files directory under meta directory
    if (sk::logger::FileLogger::Instance().IsEnabled()) {
        std::string binPath = CreateSkMetaDirectory(model);
        if (!DumpKernelBinaries(graph, binPath)) {
            SK_LOGW("Failed to dump kernel binaries: %s/bin_files", binPath.c_str());
            // Not a fatal error, continue
        }
    }
    SK_LOGI("End aclskOptimize");
    ret = DumpGraphJson(model, metaDir, deviceId, "sk_graph_rts_after");
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return ACL_SUCCESS;
}

aclError aclskScopeBegin(const char* scopeName, aclrtStream stream) {
    if (scopeName != nullptr && scopeName[0] == '\0') {
        SK_LOGE("Invalid scopeName: name is empty.");
        return ACL_ERROR_INVALID_PARAM;
    }
    return LaunchScopeKernel(scopeName, stream, true);
}

aclError aclskScopeEnd(const char* scopeName, aclrtStream stream) {
    if (scopeName != nullptr && scopeName[0] == '\0') {
        SK_LOGE("Invalid scopeName: name is empty.");
        return ACL_ERROR_INVALID_PARAM;
    }
    return LaunchScopeKernel(scopeName, stream, false);
}

#ifdef __cplusplus
}
#endif
