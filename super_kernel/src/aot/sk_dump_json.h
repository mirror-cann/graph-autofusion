/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for the details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNEsS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_dump_json.h
 * \brief Declaration for SuperKernel graph JSON dump utilities
 */

#ifndef SK_DUMP_JSON_H
#define SK_DUMP_JSON_H

#include <string>
#include <vector>
#include "sk_scope_info.h"

// Forward declarations
class SuperKernelGraph;
class SkTask;
class SuperKernelOptionsManager;

/**
 * @brief Dump SuperKernel graph node information to JSON file
 *
 * Outputs all nodes' detailed information after initialization to a JSON file
 * for debugging and analysis purposes.
 *
 * @param graph Reference to the SuperKernelGraph
 * @return true if dump successful, false otherwise
 */
bool DumpGraphNodesToJson(const SuperKernelGraph& graph);

/**
 * @brief Dump SkTask (AIC/AIV task queue) information to JSON file
 *
 * Outputs the AIC and AIV task queue information to a JSON file
 * for debugging and analysis purposes.
 *
 * @param graph Reference to the SuperKernelGraph
 * @param aicTask Reference to the AIC SkTask
 * @param aivTask Reference to the AIV SkTask
 * @return true if dump successful, false otherwise
 */
bool DumpSkTaskQueueToJson(const SuperKernelGraph& graph, const SkTask& aicTask, const SkTask& aivTask);

/**
 * @brief Dump fused (after fusion) SuperKernel graph to JSON file with nested structure
 *
 * Outputs the graph after fusion in a nested format where superkernel scopes are
 * represented as nested objects containing their constituent nodes.
 * Format: Non-fused nodes are output as flat entries, fused scopes are output as
 * nested entries containing all their member nodes.
 *
 * Example:
 *   Before fusion: node1, node2, node3, node4, node5, node6, node7, node8, node9
 *   After fusion:  node3-node6 fused into superkernel scope1
 *   JSON output:
 *   {
 *     "nodes": [
 *       {"nodeId": 1, ...},
 *       {"nodeId": 2, ...},
 *       {
 *         "scopeId": 1,
 *         "scopeName": "scope1",
 *         "nodeIds": [3, 4, 5, 6],
 *         "nodes": [...]
 *       },
 *       {"nodeId": 7, ...},
 *       ...
 *     ]
 *   }
 *
 * @param graph Reference to the SuperKernelGraph
 * @param scopeInfos Reference to the vector of SuperKernelScopeInfo containing fusion results
 * @return true if dump successful, false otherwise
 */
bool DumpFusedGraphToJson(const SuperKernelGraph& graph, const std::vector<SuperKernelScopeInfo>& scopeInfos);

/**
 * @brief Dump raw task information from modelRI to JSON file
 *
 * Directly retrieves task information from modelRI using the following interfaces:
 * - aclmdlRIGetStreams: Get all streams
 * - aclmdlRIGetTasksByStream: Get tasks in each stream
 * - aclmdlRITaskGetType: Get task type
 * - aclrtStreamGetId: Get stream ID
 * - aclmdlRITaskGetParams: Get task parameters
 *
 * The output JSON contains:
 * - Total number of streams and tasks
 * - For each stream: stream index, stream ID, task count
 * - For each task: nodeId, task type, stream info, and type-specific parameters
 *
 * @param modelRI The model RI handle
 * @param fileName Custom filename for the output JSON file (without .json suffix)
 * @param deviceId The device ID
 * @return true if dump successful, false otherwise
 */
bool DumpModelRITasksToJson(aclmdlRI modelRI, const std::string& fileName, int32_t deviceId);

/**
 * @brief Dump raw task information from modelRI to JSON file with options
 *
 * Same as DumpModelRITasksToJson, but also includes options information in the JSON output.
 *
 * @param modelRI The model RI handle
 * @param fileName Custom filename for the output JSON file (without .json suffix)
 * @param deviceId The device ID
 * @param optsMgr Reference to SuperKernelOptionsManager containing parsed options
 * @return true if dump successful, false otherwise
 */
bool DumpModelRITasksToJsonWithOpts(aclmdlRI modelRI, const std::string& fileName,
                                    int32_t deviceId, const SuperKernelOptionsManager& optsMgr);

/**
 * @brief Print original scopes before fusion to current log context
 *
 * Iterates through original scope infos and prints scopeId, scopeNames, totalNodes,
 * and nodeIds in batches.
 *
 * @param graph Reference to the SuperKernelGraph
 */
void PrintOriginalScopes(const SuperKernelGraph& graph);

/**
 * @brief Print fused scopes after fusion to current log context
 *
 * Iterates through processed scope infos, collects fused nodes (excluding scope nodes),
 * traces root break info, and prints fusionStatus, breakReason, parentScopeId, failReason,
 * and fusedNodeIds in batches.
 *
 * @param graph Reference to the SuperKernelGraph
 * @param processedScopeInfos Reference to the vector of processed scope info
 */
void PrintFusedScopes(const SuperKernelGraph& graph,
                      const std::vector<SuperKernelScopeInfo>& processedScopeInfos);

#endif // SK_DUMP_JSON_H
