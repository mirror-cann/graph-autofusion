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
#include <unordered_map>
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
 * @brief Convert SkTask (AIC/AIV task queue) to JSON object
 *
 * Converts the AIC and AIV task queue information to a JSON object
 * for later aggregation and file writing.
 *
 * @param aicTask Reference to the AIC SkTask
 * @param aivTask Reference to the AIV SkTask
 * @param scopeId Scope ID for this task queue
 * @return JSON object representing the task queue
 */
Json SkTaskToQueueJson(const SkTask& aicTask, const SkTask& aivTask, uint16_t scopeId);

/**
 * @brief Dump all task queues to a single JSON file
 *
 * Aggregates all scope task queue JSON objects into a single file.
 *
 * @param graph Reference to the SuperKernelGraph
 * @param taskQueueJsons Map of scopeId -> task queue JSON
 * @return true if dump successful, false otherwise
 */
bool DumpAllTaskQueuesToJson(const SuperKernelGraph& graph,
                             const std::unordered_map<std::string, Json>& taskQueueJsons);

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

/**
 * @brief Dump graph to test JSON file (test.json or test_2.json)
 *
 * Creates a temporary graph from modelRI, initializes it, and dumps to JSON file.
 * Only dumps when FileLogger is enabled (ASCEND_OP_COMPILE_SAVE_KERNEL_META is set).
 *
 * @param model Model RI handle
 * @param opts SuperKernel options manager
 * @param metaDir Meta directory path
 * @param filename Filename without .json suffix (e.g., "test" or "test_2")
 * @return true if dump successful or skipped, false otherwise
 */
bool DumpRawTaskJson(aclmdlRI model, const SuperKernelOptionsManager& opts, const std::string& metaDir, const std::string& filename);

#endif // SK_DUMP_JSON_H
