#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Scope 可视化

当前使用库驱动方式可视化 scope/update 关系：
  1. scope/graph library 模式: 使用 scope-library.json + graph-library.json 构建 Scope 可视化
"""

import os
import re
import sys
import json
import argparse
from collections import defaultdict, OrderedDict

from sk_visualizer_shared import (
    COMMON_DETAIL_TABLE_CSS,
    COMMON_GRAPH_FRAME_CSS,
    COMMON_GRAPH_TOOLBAR_CSS,
    COMMON_KEYBOARD_NAV_JS,
    COMMON_SCOPE_SECTION_CSS,
    COMMON_SVG_VIEWPORT_JS,
    COMMON_TABLE_CONTROLS_CSS,
    COMMON_TOOLBAR_CSS,
    COMMON_TOOLTIP_CSS,
    SVG_EDGE_TOKENS,
    SVG_LANE_TOKENS,
    SVG_NODE_TOKENS,
    make_scope_explainer_box,
    make_scope_explainer_line,
    make_scope_explainer_text,
    render_detail_table_panel,
    render_graph_index_chip,
    render_graph_nav,
    render_graph_toolbar,
    render_standard_table_pager_html,
    render_standard_table_tools_html,
    VISUAL_FONT_MONO_STACK,
    build_visualizer_styles,
    render_scope_section_block,
    render_page_header,
    render_view_nav,
)


def _emit(message: object = "", *, file=None, end: str = "\n") -> None:
    stream = sys.stdout if file is None else file
    stream.write(f"{message}{end}")


# ============================================================================
# 数据结构
# ============================================================================


class SkNode:
    """表示日志中的单个节点"""

    __slots__ = [
        "node_id",
        "stream_id",
        "stream_idx_in_graph",
        "node_idx_in_stream",
        "node_type",
        "event_id",
        "event_flag",
        "kernel_name",
        "scope_id",
        "round_idx",
        "index_in_list",
        "kernel_type",
        "task_ratio",
        "num_blocks",
        "cube_num",
        "vec_num",
        "resolved_num",
        "scope_flags",
        "is_fusible",
        "update_type",
        "update_addr",
        "update_value",
        "update_flag",
        "update_op_info_ptr",
        "update_op_info_size",
        "update_func_handle",
        "update_args",
        "update_args_size",
        "update_num_blocks",
        "identity_kind",
        "custom_instance_key",
        "raw_node_id",
        "queue_name",
        "inherited_notify_node_ids",
        "inherited_wait_node_ids",
        "inherited_reset_node_ids",
        "inherited_notify_addrs",
        "inherited_wait_addrs",
        "inherited_reset_addrs",
        "inherited_write_facts",
        "inherited_wait_facts",
    ]

    def __init__(
        self,
        node_id,
        stream_id,
        stream_idx_in_graph,
        node_idx_in_stream,
        node_type,
        event_id=None,
        event_flag=None,
        kernel_name=None,
        scope_id=None,
        round_idx=None,
        index_in_list=None,
        kernel_type="",
        task_ratio=(0, 0),
        num_blocks=0,
        cube_num=0,
        vec_num=0,
        resolved_num=0,
        scope_flags="",
        is_fusible=True,
    ):
        self.node_id = node_id
        self.stream_id = stream_id
        self.stream_idx_in_graph = stream_idx_in_graph
        self.node_idx_in_stream = node_idx_in_stream
        self.node_type = node_type
        self.event_id = event_id
        self.event_flag = event_flag
        self.kernel_name = kernel_name
        self.scope_id = scope_id
        self.round_idx = round_idx
        self.index_in_list = index_in_list
        self.kernel_type = kernel_type
        self.task_ratio = task_ratio
        self.num_blocks = num_blocks
        self.cube_num = cube_num
        self.vec_num = vec_num
        self.resolved_num = resolved_num
        self.scope_flags = scope_flags
        self.is_fusible = is_fusible
        self.update_type = ""
        self.update_addr = ""
        self.update_value = ""
        self.update_flag = ""
        self.update_op_info_ptr = ""
        self.update_op_info_size = 0
        self.update_func_handle = ""
        self.update_args = ""
        self.update_args_size = 0
        self.update_num_blocks = 0
        self.identity_kind = "graph_node"
        self.custom_instance_key = ""
        self.raw_node_id = None
        self.queue_name = ""
        self.inherited_notify_node_ids = []
        self.inherited_wait_node_ids = []
        self.inherited_reset_node_ids = []
        self.inherited_notify_addrs = []
        self.inherited_wait_addrs = []
        self.inherited_reset_addrs = []
        self.inherited_write_facts = []
        self.inherited_wait_facts = []

    @property
    def display_name(self):
        """用于表格展示的名称"""
        if self.node_type == "Kernel":
            name = (self.kernel_name or "").replace("static_kernel_", "")
            # 去掉末尾的 _d0, __kernel0 等
            name = re.sub(r"_d\d+(_?\w*)$", "", name)
            return name if len(name) <= 40 else name[:37] + "..."
        short = self.node_type.replace("Event", "")
        eid = (self.event_id or "N/A")[-8:]
        return f"{short}(0x{eid})"

    @property
    def short_kernel_name(self):
        """截断的 kernel 名称用于 SVG 节点显示"""
        name = (self.kernel_name or "").replace("static_kernel_", "")
        name = re.sub(r"_d\d+(_?\w*)$", "", name)
        return name if len(name) <= 17 else name[:15] + ".."

    def to_dict(self):
        return {
            "id": self.node_id,
            "stream_id": self.stream_id,
            "stream_idx": self.stream_idx_in_graph,
            "node_idx_in_stream": self.node_idx_in_stream,
            "type": self.node_type,
            "event_id": self.event_id,
            "event_flag": self.event_flag,
            "kernel_name": self.kernel_name,
            "scope_id": self.scope_id,
            "display_name": self.display_name,
            "kernel_type": self.kernel_type,
            "task_ratio": list(self.task_ratio) if self.task_ratio else [0, 0],
            "num_blocks": self.num_blocks,
            "cube_num": self.cube_num,
            "vec_num": self.vec_num,
            "resolved_num": self.resolved_num,
            "scope_flags": self.scope_flags,
            "is_fusible": self.is_fusible,
            "update_type": self.update_type,
            "update_addr": self.update_addr,
            "update_value": self.update_value,
            "update_flag": self.update_flag,
            "update_op_info_ptr": self.update_op_info_ptr,
            "update_op_info_size": self.update_op_info_size,
            "update_func_handle": self.update_func_handle,
            "update_args": self.update_args,
            "update_args_size": self.update_args_size,
            "update_num_blocks": self.update_num_blocks,
            "identity_kind": self.identity_kind,
            "custom_instance_key": self.custom_instance_key,
            "raw_node_id": self.raw_node_id,
            "queue_name": self.queue_name,
            "inherited_notify_node_ids": self.inherited_notify_node_ids,
            "inherited_wait_node_ids": self.inherited_wait_node_ids,
            "inherited_reset_node_ids": self.inherited_reset_node_ids,
            "inherited_notify_addrs": self.inherited_notify_addrs,
            "inherited_wait_addrs": self.inherited_wait_addrs,
            "inherited_reset_addrs": self.inherited_reset_addrs,
            "inherited_write_facts": self.inherited_write_facts,
            "inherited_wait_facts": self.inherited_wait_facts,
        }


class SkScope:
    """表示一个 Scope"""

    def __init__(self, scope_id, num_nodes, num_streams, bit_flags, names, round_idx):
        self.scope_id = scope_id
        self.num_nodes = num_nodes
        self.num_streams = num_streams
        self.bit_flags = bit_flags
        self.names = names
        self.round_idx = round_idx
        self.nodes = []

    def to_dict(self):
        return {
            "id": self.scope_id,
            "num_nodes": self.num_nodes,
            "num_streams": self.num_streams,
            "bit_flags": self.bit_flags,
            "names": self.names,
        }


class SkScopeRound:
    """表示一轮 Scope 切分结果"""

    def __init__(self, round_idx, line_content=""):
        self.round_idx = round_idx
        self.line_content = line_content
        self.scopes = []
        self.all_nodes = {}  # node_id -> SkNode
        self.source_file = ""
        self.source_round_idx = -1
        self.model_instances = []
        self.custom_update_nodes = []
        self.synthesized_custom_annotations = []
        self.update_edges = []
        self.update_summary = {}


class SkEdge:
    """表示节点间的连边"""

    __slots__ = ["src_id", "dst_id", "edge_type", "event_id"]

    def __init__(self, src_id, dst_id, edge_type, event_id=None):
        self.src_id = src_id
        self.dst_id = dst_id
        self.edge_type = edge_type  # 'stream' 或 'event'
        self.event_id = event_id


class ScopeLibrarySource:
    """Resolve scope/graph library JSON inputs for scope graph generation."""

    def __init__(self, scope_library_path, graph_library_path, mode="scope"):
        self.scope_library_path = scope_library_path
        self.graph_library_path = graph_library_path
        self.mode = mode

    def collect(self):
        if not os.path.exists(self.scope_library_path):
            raise FileNotFoundError(self.scope_library_path)
        if not os.path.exists(self.graph_library_path):
            raise FileNotFoundError(self.graph_library_path)
        return [self.scope_library_path, self.graph_library_path]


class ScopeGraphModel:
    """Stable object wrapper around parsed scope rounds and metadata."""

    def __init__(self, rounds, log_files, event_colors, init_nodes=None, mode="scope"):
        self.rounds = rounds
        self.log_files = log_files
        self.event_colors = event_colors
        self.init_nodes = init_nodes or {}
        self.mode = mode

    @classmethod
    def from_libraries(cls, source):
        if source.mode != "scope":
            raise ValueError(
                "ScopeGraphModel.from_libraries currently only supports scope mode."
            )

        scope_library_path, graph_library_path = source.collect()
        with open(scope_library_path, "r", encoding="utf-8") as handle:
            scope_library = json.load(handle)
        with open(graph_library_path, "r", encoding="utf-8") as handle:
            graph_library = json.load(handle)

        rounds = build_rounds_from_libraries(scope_library, graph_library)
        event_colors = assign_event_colors(rounds)
        return cls(
            rounds,
            [scope_library_path, graph_library_path],
            event_colors,
            {},
            source.mode,
        )


class ScopeGraphRenderer:
    """Render the existing interactive scope graph without rewriting its UI logic."""

    def __init__(self, output_path):
        self.output_path = output_path

    def render_html(self, model):
        return generate_html(
            model.rounds,
            model.event_colors,
            model.log_files,
            self.output_path,
            init_nodes=model.init_nodes,
            mode=model.mode,
        )


def _normalize_library_node_type(node_entry):
    node_type = str(node_entry.get("node_type") or "").upper()
    event_type = node_entry.get("event_type")
    if node_type == "KERNEL":
        return "Kernel"
    if event_type in {
        "EventNotify",
        "EventWait",
        "EventReset",
        "MemoryWrite",
        "MemoryWait",
    }:
        return event_type
    mapping = {
        "EVENT_NOTIFY": "EventNotify",
        "EVENT_WAIT": "EventWait",
        "EVENT_RESET": "EventReset",
        "MEMORY_WRITE": "MemoryWrite",
        "MEMORY_WAIT": "MemoryWait",
        "NOTIFY": "EventNotify",
        "WAIT": "EventWait",
        "RESET": "EventReset",
    }
    return mapping.get(node_type, "Unknown")


def _update_type(row):
    if row is None:
        return ""
    return str(row.get("type") or "").strip().upper()


def build_rounds_from_libraries(scope_library, graph_library):
    """Build one scope graph round per final scope from scope/graph libraries."""

    node_library = graph_library.get("node_library", {})
    node_update_rows = graph_library.get("node_update_registry", {}).get("rows", [])
    node_by_id = {
        node.get("node_id"): node
        for node in node_library.get("nodes", [])
        if isinstance(node.get("node_id"), int)
    }
    node_update_by_id = {
        row.get("node_id"): row
        for row in node_update_rows
        if isinstance(row.get("node_id"), int)
    }
    event_addr_by_node_id, event_addr_facts = _build_event_addr_facts(
        {
            "scope_library": scope_library,
            "graph_library": graph_library,
        }
    )
    effective_node_update_rows = []
    for row in node_update_rows:
        merged = dict(row)
        node_id = merged.get("node_id")
        merged_type = _update_type(merged)
        if (
            isinstance(node_id, int)
            and node_id in event_addr_by_node_id
            and merged_type in {"VALUE_WRITE", "VALUE_WAIT"}
        ):
            merged["effective_addr"] = event_addr_by_node_id[node_id]
        else:
            merged["effective_addr"] = merged.get("addr")
        effective_node_update_rows.append(merged)
    effective_node_update_by_id = {
        row.get("node_id"): row
        for row in effective_node_update_rows
        if isinstance(row.get("node_id"), int)
    }
    event_roles_by_node_id = defaultdict(
        lambda: {
            "notify_node_ids": set(),
            "wait_node_ids": set(),
            "notify_addrs": set(),
            "wait_addrs": set(),
        }
    )
    for addr, facts in event_addr_facts.items():
        for node_id in facts.get("notify_node_ids", []):
            role = event_roles_by_node_id[node_id]
            role["notify_node_ids"].add(node_id)
            role["notify_addrs"].add(addr)
        for node_id in facts.get("wait_node_ids", []):
            role = event_roles_by_node_id[node_id]
            role["wait_node_ids"].add(node_id)
            role["wait_addrs"].add(addr)
        for row in facts.get("write_rows", []):
            node_id = row.get("node_id")
            if not isinstance(node_id, int):
                continue
            role = event_roles_by_node_id[node_id]
            role["notify_node_ids"].add(node_id)
            role["notify_addrs"].add(addr)
        for row in facts.get("wait_rows", []):
            node_id = row.get("node_id")
            if not isinstance(node_id, int):
                continue
            role = event_roles_by_node_id[node_id]
            role["wait_node_ids"].add(node_id)
            role["wait_addrs"].add(addr)

    fused_functions = scope_library.get("fused_library", {}).get("functions", [])
    device_sections = scope_library.get("device_task_library", {}).get("sections", [])
    device_section_by_scope_name = {
        str(section.get("scope_name") or ""): section
        for section in device_sections
        if str(section.get("scope_name") or "")
    }

    rounds = []
    for round_idx, scope in enumerate(scope_library.get("scopes", [])):
        scope_id = scope.get("scope_id")
        round_data = SkScopeRound(round_idx, f"scope {scope_id}")
        round_data.source_file = os.path.basename(
            scope_library.get("path") or "scope-library.json"
        )
        round_data.source_round_idx = 1
        round_data.model_instances = [1]

        scope_name = (
            str(scope.get("scope_names", [])[0] if scope.get("scope_names") else "")
            or ""
        )
        scope_obj = SkScope(
            scope_id=scope_id,
            num_nodes=scope.get("node_count", 0),
            num_streams=scope.get("stream_count", 0),
            bit_flags=scope.get("scope_bit_flags"),
            names=[scope_name] if scope_name else [],
            round_idx=round_idx,
        )
        round_data.scopes.append(scope_obj)
        update_payload = (
            scope.get("update") if isinstance(scope.get("update"), dict) else {}
        )
        graph_backed_updates = (
            update_payload.get("graph_backed_updates", [])
            if isinstance(update_payload, dict)
            else []
        )
        graph_backed_update_by_id = {
            item.get("node_id"): item
            for item in graph_backed_updates
            if isinstance(item, dict) and isinstance(item.get("node_id"), int)
        }

        def _build_scope_node(node_id, node_scope_id, current_round_idx=round_idx):
            node_entry = node_by_id.get(node_id)
            if not node_entry:
                return None
            kernel_info = _parse_kernel_infos_from_detail(node_entry.get("detail", ""))
            node = SkNode(
                node_id=node_entry.get("node_id"),
                stream_id=node_entry.get("stream_id", -1),
                stream_idx_in_graph=node_entry.get("stream_idx_in_graph", -1),
                node_idx_in_stream=node_entry.get("node_idx_in_stream", -1),
                node_type=_normalize_library_node_type(node_entry),
                event_id=node_entry.get("event_id"),
                event_flag=node_entry.get("event_flag"),
                kernel_name=node_entry.get("func_name"),
                scope_id=node_scope_id,
                round_idx=current_round_idx,
                index_in_list=0,
                kernel_type=node_entry.get("kernel_type")
                or kernel_info.get("kernel_type", ""),
                task_ratio=kernel_info.get("task_ratio", (0, 0)),
                num_blocks=kernel_info.get("num_blocks", 0),
                cube_num=kernel_info.get("cube_num", 0),
                vec_num=kernel_info.get("vec_num", 0),
                resolved_num=kernel_info.get("resolved_num", 0),
            )
            node_update = node_update_by_id.get(node.node_id)
            if node_update:
                node.update_type = _update_type(node_update)
                node.update_addr = str(node_update.get("addr") or "")
                node.update_value = str(node_update.get("value") or "")
                node.update_flag = str(node_update.get("flag") or "")
                node.update_op_info_ptr = str(node_update.get("op_info_ptr") or "")
                node.update_op_info_size = int(node_update.get("op_info_size") or 0)
                node.update_func_handle = str(node_update.get("func_handle") or "")
                node.update_args = str(node_update.get("args") or "")
                node.update_args_size = int(node_update.get("args_size") or 0)
                node.update_num_blocks = int(node_update.get("num_blocks") or 0)
            return node

        scope_nodes = []
        for node_id in scope.get("node_ids", []):
            node = _build_scope_node(node_id, scope_id)
            if not node:
                continue
            graph_backed_update = graph_backed_update_by_id.get(node.node_id)
            if graph_backed_update:
                node.update_type = str(
                    graph_backed_update.get("update_type") or node.update_type or ""
                )
                node.update_addr = str(graph_backed_update.get("addr") or "")
                node.update_value = str(graph_backed_update.get("value") or "")
                node.update_flag = str(graph_backed_update.get("flag") or "")
                node.update_op_info_ptr = str(
                    graph_backed_update.get("op_info_ptr") or ""
                )
                node.update_op_info_size = int(
                    graph_backed_update.get("op_info_size") or 0
                )
                node.update_func_handle = str(
                    graph_backed_update.get("func_handle") or ""
                )
                node.update_args = str(graph_backed_update.get("args") or "")
                node.update_args_size = int(graph_backed_update.get("args_size") or 0)
                node.update_num_blocks = int(graph_backed_update.get("num_blocks") or 0)
            scope_nodes.append(node)
            round_data.all_nodes[node.node_id] = node

        scope_nodes.sort(
            key=lambda item: (item.stream_id, item.node_idx_in_stream, item.node_id)
        )
        for index, node in enumerate(scope_nodes):
            node.index_in_list = index
        scope_obj.nodes = scope_nodes
        synthesized_custom_annotations = list(
            update_payload.get("synthesized_custom_nodes", [])
            if isinstance(update_payload, dict)
            else []
        )
        round_data.custom_update_nodes = []
        round_data.synthesized_custom_annotations = synthesized_custom_annotations

        scope_node_ids = {node.node_id for node in scope_nodes}
        fused_function = _match_fused_function(scope_node_ids, fused_functions)
        fused_scope_name = str(fused_function.get("scope_name") or "")
        if fused_scope_name:
            scope_obj.names = [fused_scope_name]
        elif not scope_obj.names:
            section_by_scope_id = next(
                (
                    str(section.get("scope_name") or "")
                    for section in device_sections
                    if section.get("scope_id") == scope_id
                    and str(section.get("scope_name") or "")
                ),
                "",
            )
            if section_by_scope_id:
                scope_obj.names = [section_by_scope_id]
        fused_member_ids = [
            node_id
            for node_id in fused_function.get("node_ids", [])
            if node_id in scope_node_ids
        ]
        carrier_node = None
        if fused_member_ids:
            kernel_update_ids = [
                node_id
                for node_id in fused_member_ids
                if _update_type(node_update_by_id.get(node_id, {})) == "KERNEL"
            ]
            carrier_candidates = kernel_update_ids or [
                node_id
                for node_id in fused_member_ids
                if str(node_by_id.get(node_id, {}).get("node_type") or "").upper()
                == "KERNEL"
            ]
            if carrier_candidates:
                carrier_node = round_data.all_nodes.get(carrier_candidates[0])
        if carrier_node:
            section = device_section_by_scope_name.get(
                str(fused_function.get("scope_name") or ""), {}
            )
            ordinal_to_node_id = {
                item.get("ordinal"): item.get("node_id")
                for item in fused_function.get("node_details", [])
                if isinstance(item.get("ordinal"), int)
                and isinstance(item.get("node_id"), int)
            }
            scope_node_by_id = {node.node_id: node for node in scope_nodes}
            inherited_notify_node_ids = set()
            inherited_wait_node_ids = set()
            inherited_notify_addrs = set()
            inherited_wait_addrs = set()
            inherited_write_facts = OrderedDict()
            inherited_wait_facts = OrderedDict()
            section_tasks = []
            for queue_name in ("AIC", "AIV"):
                section_tasks.extend(section.get("queues", {}).get(queue_name, []))
            if section_tasks:
                for task in section_tasks:
                    task_type = str(task.get("task_type") or "").strip().upper()
                    if task_type not in {"EVENT_NOTIFY", "EVENT_WAIT", "EVENT_RESET"}:
                        continue
                    task_addr = str(task.get("args") or "").strip()
                    if not task_addr:
                        continue
                    node_id = ordinal_to_node_id.get(task.get("task_index"))
                    node_obj = (
                        scope_node_by_id.get(node_id)
                        if isinstance(node_id, int)
                        else None
                    )
                    effective_row = (
                        effective_node_update_by_id.get(node_id)
                        if isinstance(node_id, int)
                        else None
                    )
                    if task_type == "EVENT_NOTIFY":
                        fact = inherited_write_facts.setdefault(
                            task_addr, {"addr": task_addr, "values": [], "node_ids": []}
                        )
                        if (
                            node_obj
                            and effective_row
                            and node_obj.node_id != carrier_node.node_id
                            and _update_type(effective_row) == "VALUE_WRITE"
                            and str(effective_row.get("effective_addr") or "")
                            == task_addr
                        ):
                            if node_id not in fact["node_ids"]:
                                fact["node_ids"].append(node_id)
                            val = str(effective_row.get("value") or "")
                            if val and val not in fact["values"]:
                                fact["values"].append(val)
                            inherited_notify_node_ids.add(node_id)
                        inherited_notify_addrs.add(task_addr)
                    elif task_type == "EVENT_WAIT":
                        fact = inherited_wait_facts.setdefault(
                            task_addr,
                            {
                                "addr": task_addr,
                                "values": [],
                                "flags": [],
                                "rules": [],
                                "node_ids": [],
                            },
                        )
                        if (
                            node_obj
                            and effective_row
                            and node_obj.node_id != carrier_node.node_id
                            and _update_type(effective_row) == "VALUE_WAIT"
                            and str(effective_row.get("effective_addr") or "")
                            == task_addr
                        ):
                            if node_id not in fact["node_ids"]:
                                fact["node_ids"].append(node_id)
                            val = str(effective_row.get("value") or "")
                            flg = str(effective_row.get("flag") or "")
                            if val and val not in fact["values"]:
                                fact["values"].append(val)
                            if flg and flg not in fact["flags"]:
                                fact["flags"].append(flg)
                            rule = _wait_rule_text(flg)
                            if rule and rule not in fact["rules"]:
                                fact["rules"].append(rule)
                            inherited_wait_node_ids.add(node_id)
                        inherited_wait_addrs.add(task_addr)
            else:
                for member_node_id in fused_member_ids:
                    if member_node_id == carrier_node.node_id:
                        continue
                    roles = event_roles_by_node_id.get(member_node_id)
                    if not roles:
                        continue
                    inherited_notify_node_ids.update(roles["notify_node_ids"])
                    inherited_wait_node_ids.update(roles["wait_node_ids"])
                    inherited_notify_addrs.update(roles["notify_addrs"])
                    inherited_wait_addrs.update(roles["wait_addrs"])
            carrier_node.inherited_notify_node_ids = sorted(inherited_notify_node_ids)
            carrier_node.inherited_wait_node_ids = sorted(inherited_wait_node_ids)
            carrier_node.inherited_reset_node_ids = []
            carrier_node.inherited_notify_addrs = sorted(inherited_notify_addrs)
            carrier_node.inherited_wait_addrs = sorted(inherited_wait_addrs)
            carrier_node.inherited_reset_addrs = []
            carrier_node.inherited_write_facts = list(inherited_write_facts.values())
            carrier_node.inherited_wait_facts = list(inherited_wait_facts.values())

        update_edges = []
        seen_update_edges = set()
        notify_wait_count = 0
        updated_node_ids = {node.node_id for node in scope_nodes if node.update_type}
        carrier_notify_addrs = {}
        for node in scope_nodes:
            if node.inherited_notify_addrs:
                carrier_notify_addrs[node.node_id] = set(node.inherited_notify_addrs)
            for member_id in node.inherited_notify_node_ids:
                if member_id not in scope_node_ids or member_id == node.node_id:
                    continue
                edge_key = ("sk_member_write", member_id, node.node_id)
                if edge_key in seen_update_edges:
                    continue
                seen_update_edges.add(edge_key)
                update_edges.append(SkEdge(member_id, node.node_id, "update_sk_write"))
            for member_id in node.inherited_wait_node_ids:
                if member_id not in scope_node_ids or member_id == node.node_id:
                    continue
                edge_key = ("sk_member_wait", node.node_id, member_id)
                if edge_key in seen_update_edges:
                    continue
                seen_update_edges.add(edge_key)
                update_edges.append(SkEdge(node.node_id, member_id, "update_sk_wait"))
            for member_id in node.inherited_reset_node_ids:
                if member_id not in scope_node_ids or member_id == node.node_id:
                    continue
                edge_key = ("sk_member_write", member_id, node.node_id)
                if edge_key in seen_update_edges:
                    continue
                seen_update_edges.add(edge_key)
                update_edges.append(SkEdge(member_id, node.node_id, "update_sk_write"))

        for addr, facts in event_addr_facts.items():
            notify_rows = list(facts.get("write_rows", []))
            wait_rows = list(facts.get("wait_rows", []))

            for carrier_node_id, inherited_addrs in carrier_notify_addrs.items():
                if addr not in inherited_addrs:
                    continue
                for wait_row in wait_rows:
                    dst_id = wait_row.get("node_id")
                    if (
                        carrier_node_id not in scope_node_ids
                        or dst_id not in scope_node_ids
                    ):
                        continue
                    if carrier_node_id == dst_id:
                        continue
                    edge_key = ("notify_wait", carrier_node_id, dst_id, addr)
                    if edge_key in seen_update_edges:
                        continue
                    seen_update_edges.add(edge_key)
                    notify_wait_count += 1
                    update_edges.append(
                        SkEdge(carrier_node_id, dst_id, "update_notify_wait", addr)
                    )

            for write_row in notify_rows:
                src_id = write_row.get("node_id")
                for wait_row in wait_rows:
                    if not _write_is_notify_for_wait(write_row, wait_row):
                        continue
                    dst_id = wait_row.get("node_id")
                    if src_id not in scope_node_ids or dst_id not in scope_node_ids:
                        continue
                    edge_key = ("notify_wait", src_id, dst_id, addr)
                    if edge_key in seen_update_edges:
                        continue
                    seen_update_edges.add(edge_key)
                    notify_wait_count += 1
                    update_edges.append(
                        SkEdge(src_id, dst_id, "update_notify_wait", addr)
                    )

        round_data.update_edges = update_edges
        round_data.update_summary = {
            "updated_node_count": len(updated_node_ids),
            "synthesized_custom_count": len(synthesized_custom_annotations),
            "notify_wait_edge_count": notify_wait_count,
            "wait_reset_edge_count": 0,
            "addr_count": len(
                {edge.event_id for edge in update_edges if edge.event_id}
            ),
        }
        rounds.append(round_data)

    return rounds


# ============================================================================
# 正则表达式
# ============================================================================

# KernelInfos 字段提取
RE_KI_FUNCNAME = re.compile(r"funcName:([^,}]+)")
RE_KI_KERNELTYPE = re.compile(r"kernelType:(\w+)")
RE_KI_TASKRATIO = re.compile(r"taskRatio:\[([^,]+),([^\]]+)\]")
RE_KI_NUMBLOCKS = re.compile(r"numBlocks:(\d+)")
RE_KI_CUBENUM = re.compile(r"cubeNum:(\d+)")
RE_KI_VECNUM = re.compile(r"vecNum:(\d+)")
RE_KI_RESOLVEDNUM = re.compile(r"resolvedNum:(\d+)")

# ============================================================================
# 日志解析
# ============================================================================


def _parse_kernel_infos(ki_str):
    """解析 KernelInfos 字符串，返回字段字典"""
    if not ki_str:
        return {}
    result = {}
    m = RE_KI_FUNCNAME.search(ki_str)
    if m:
        result["kernel_name"] = m.group(1).strip()
    m = RE_KI_KERNELTYPE.search(ki_str)
    if m:
        result["kernel_type"] = m.group(1).strip()
    m = RE_KI_TASKRATIO.search(ki_str)
    if m:
        result["task_ratio"] = (int(m.group(1).strip()), int(m.group(2).strip()))
    m = RE_KI_NUMBLOCKS.search(ki_str)
    if m:
        result["num_blocks"] = int(m.group(1))
    m = RE_KI_CUBENUM.search(ki_str)
    if m:
        result["cube_num"] = int(m.group(1))
    m = RE_KI_VECNUM.search(ki_str)
    if m:
        result["vec_num"] = int(m.group(1))
    m = RE_KI_RESOLVEDNUM.search(ki_str)
    if m:
        result["resolved_num"] = int(m.group(1))
    return result


def _parse_kernel_infos_from_detail(detail_str):
    if not detail_str:
        return {}
    match = re.search(r"KernelInfos\{([^}]*)\}", detail_str)
    if not match:
        return {}
    return _parse_kernel_infos(match.group(1))


def _safe_parse_int(value):
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text)
    except Exception:
        return None


def _wait_rule_text(flag_value):
    flag = _safe_parse_int(flag_value)
    if flag == 0:
        return "write >= wait"
    if flag == 1:
        return "write == wait"
    if flag == 2:
        return "(write & wait) != 0"
    if flag == 3:
        return "(~(write | wait)) != 0"
    if flag_value:
        return str(flag_value)
    return ""


def _match_fused_function(scope_node_ids, fused_functions):
    best = {}
    best_overlap = -1
    for func in fused_functions:
        node_ids = set(func.get("node_ids", []))
        overlap = len(scope_node_ids & node_ids)
        if overlap > best_overlap:
            best = func
            best_overlap = overlap
    return best


def _build_event_addr_facts(report):
    scope_library = (
        report.get("scope_library", {})
        if isinstance(report.get("scope_library"), dict)
        else {}
    )
    graph_library = (
        report.get("graph_library", {})
        if isinstance(report.get("graph_library"), dict)
        else {}
    )
    node_updates = graph_library.get("node_update_registry", {}).get("rows", [])

    addr_by_node_id = {}
    grouped = {}
    seen_wait_rows = set()
    seen_write_rows = set()

    update_node_lines = set()
    for scope in scope_library.get("scopes", []):
        update_payload = scope.get("update")
        if isinstance(update_payload, dict):
            update_node_lines.update(update_payload.get("node_details", []))

    explicit_addr_by_node_id = {}
    for detail in update_node_lines:
        event_addr_m = re.search(
            r"Updated (notify|wait|reset) node addrValue:\s*nodeId=(\d+),\s*addr=(0x[0-9a-fA-F]+)",
            detail,
        )
        if not event_addr_m:
            continue
        explicit_addr_by_node_id[int(event_addr_m.group(2))] = event_addr_m.group(3)

    for row in node_updates:
        node_id = row.get("node_id")
        addr = explicit_addr_by_node_id.get(node_id) or row.get("addr")
        target_type = _update_type(row)
        if (
            node_id is None
            or not addr
            or target_type not in {"VALUE_WRITE", "VALUE_WAIT"}
        ):
            continue
        addr_by_node_id[node_id] = addr
        bucket = grouped.setdefault(
            addr,
            {
                "notify_node_ids": set(),
                "wait_node_ids": set(),
                "write_rows": [],
                "wait_rows": [],
                "details": [],
            },
        )
        detail = row.get("detail")
        if detail:
            bucket["details"].append(detail)
        if target_type == "VALUE_WAIT":
            row_key = (node_id, addr, row.get("value"), row.get("flag"))
            if row_key not in seen_wait_rows:
                seen_wait_rows.add(row_key)
                bucket["wait_node_ids"].add(node_id)
                wait_row = dict(row)
                wait_row["addr"] = addr
                bucket["wait_rows"].append(wait_row)
            continue
        row_key = (node_id, addr, row.get("value"), row.get("flag"))
        if row_key not in seen_write_rows:
            seen_write_rows.add(row_key)
            write_row = dict(row)
            write_row["addr"] = addr
            bucket["write_rows"].append(write_row)

    for detail in update_node_lines:
        if "addr=" not in detail:
            continue
        event_addr_m = re.search(
            r"Updated (notify|wait|reset) node addrValue:\s*nodeId=(\d+),\s*addr=(0x[0-9a-fA-F]+)",
            detail,
        )
        if event_addr_m:
            kind = event_addr_m.group(1)
            node_id = int(event_addr_m.group(2))
            addr = event_addr_m.group(3)
            addr_by_node_id.setdefault(node_id, addr)
            bucket = grouped.setdefault(
                addr,
                {
                    "notify_node_ids": set(),
                    "wait_node_ids": set(),
                    "write_rows": [],
                    "wait_rows": [],
                    "details": [],
                },
            )
            bucket["details"].append(detail)
            if kind == "wait":
                row_key = (node_id, addr, None, None)
                if row_key not in seen_wait_rows:
                    seen_wait_rows.add(row_key)
                    bucket["wait_node_ids"].add(node_id)
                    bucket["wait_rows"].append(
                        {
                            "node_id": node_id,
                            "addr": addr,
                            "type": "VALUE_WAIT",
                            "value": None,
                            "flag": None,
                            "detail": detail,
                        }
                    )
            else:
                row_key = (node_id, addr, None, None)
                if row_key not in seen_write_rows:
                    seen_write_rows.add(row_key)
                    bucket["notify_node_ids"].add(node_id)
                    bucket["write_rows"].append(
                        {
                            "node_id": node_id,
                            "addr": addr,
                            "type": "VALUE_WRITE",
                            "value": None,
                            "flag": None,
                            "detail": detail,
                        }
                    )
            continue
        node_m = re.search(r"nodeId=(\d+)", detail)
        addr_m = re.search(r"addr=(0x[0-9a-fA-F]+)", detail)
        target_m = re.search(r"type=([A-Z_]+)", detail)
        value_m = re.search(r"value=(0x[0-9a-fA-F]+|\d+)", detail)
        flag_m = re.search(r"flag=(0x[0-9a-fA-F]+|\d+)", detail)
        if not node_m or not addr_m or not target_m:
            continue
        node_id = int(node_m.group(1))
        addr = explicit_addr_by_node_id.get(node_id) or addr_m.group(1)
        target_type = target_m.group(1)
        addr_by_node_id.setdefault(node_id, addr)
        bucket = grouped.setdefault(
            addr,
            {
                "notify_node_ids": set(),
                "wait_node_ids": set(),
                "write_rows": [],
                "wait_rows": [],
                "details": [],
            },
        )
        bucket["details"].append(detail)
        row = {
            "node_id": node_id,
            "addr": addr,
            "type": target_type,
            "value": value_m.group(1) if value_m else None,
            "flag": flag_m.group(1) if flag_m else None,
            "detail": detail,
        }
        if target_type == "VALUE_WAIT":
            row_key = (node_id, addr, row.get("value"), row.get("flag"))
            if row_key not in seen_wait_rows:
                seen_wait_rows.add(row_key)
                bucket["wait_node_ids"].add(node_id)
                bucket["wait_rows"].append(row)
        elif target_type == "VALUE_WRITE":
            row_key = (node_id, addr, row.get("value"), row.get("flag"))
            if row_key not in seen_write_rows:
                seen_write_rows.add(row_key)
                bucket["write_rows"].append(row)

    return addr_by_node_id, grouped


def _write_is_notify_for_wait(write_row, wait_row):
    write_value = _safe_parse_int(write_row.get("value"))
    wait_value = _safe_parse_int(wait_row.get("value"))
    wait_flag = _safe_parse_int(wait_row.get("flag"))
    if write_value is None or wait_value is None or wait_flag is None:
        return False
    if wait_flag == 0x0:
        return write_value >= wait_value
    if wait_flag == 0x1:
        return write_value == wait_value
    if wait_flag == 0x2:
        return (write_value & wait_value) != 0
    if wait_flag == 0x3:
        return (~(write_value | wait_value)) != 0
    return False


# ============================================================================
# 构建连边
# ============================================================================


def build_edges(round_data, extra_nodes=None):
    """
    根据节点信息构建连边:
    1. 流内连边: 同一 stream 中相邻节点（按 nodeIdxInStream 排序）
    2. 事件连边: EventNotify 与 EventWait 匹配（相同 eventId）
    3. 内存连边: MemoryWrite 与 MemoryWait 匹配（相同 eventId）

    extra_nodes: 可选的额外节点列表（如 InitNode 中解析出的 EventReset），用于扩展图
    """
    nodes = {}
    # 优先使用 round 数据中的节点
    for nid, node in round_data.all_nodes.items():
        nodes[nid] = node
    # 合并额外节点
    if extra_nodes:
        for nid, info in extra_nodes.items():
            if nid not in nodes:
                nodes[nid] = SkNode(
                    node_id=info["node_id"],
                    stream_id=info["stream_id"],
                    stream_idx_in_graph=info.get("stream_idx_in_graph", -1),
                    node_idx_in_stream=info["node_idx_in_stream"],
                    node_type=info["node_type"],
                    event_id=info.get("event_id"),
                    event_flag=info.get("event_flag"),
                    kernel_name=info.get("kernel_name"),
                    scope_id=None,  # 不在任何 scope 中
                )

    edges = []

    # 1. 流内连边
    stream_nodes = defaultdict(list)
    for node in nodes.values():
        stream_nodes[node.stream_id].append(node)
    for _, s_nodes in stream_nodes.items():
        s_nodes.sort(key=lambda n: n.node_idx_in_stream)
        for i in range(len(s_nodes) - 1):
            edges.append(SkEdge(s_nodes[i].node_id, s_nodes[i + 1].node_id, "stream"))

    # 2. 事件连边: Notify → Wait
    notify_map = {}  # event_id -> [node, ...]
    wait_map = defaultdict(list)  # event_id -> [node, ...]
    reset_map = defaultdict(list)  # event_id -> [node, ...]

    for node in nodes.values():
        if node.node_type == "EventNotify" and node.event_id:
            notify_map.setdefault(node.event_id, []).append(node)
        elif node.node_type == "EventWait" and node.event_id:
            wait_map[node.event_id].append(node)
        elif node.node_type == "EventReset" and node.event_id:
            reset_map[node.event_id].append(node)

    for eid, notifies in notify_map.items():
        for notify in notifies:
            for wait in wait_map.get(eid, []):
                edges.append(SkEdge(notify.node_id, wait.node_id, "event", eid))

    # 3. 内存连边: MemoryWrite → MemoryWait (相同 eventId)
    mw_map = defaultdict(list)
    mwait_map = defaultdict(list)

    for node in nodes.values():
        if node.node_type == "MemoryWrite" and node.event_id:
            mw_map[node.event_id].append(node)
        elif node.node_type == "MemoryWait" and node.event_id:
            mwait_map[node.event_id].append(node)

    for eid, writes in mw_map.items():
        for write in writes:
            for wait in mwait_map.get(eid, []):
                edges.append(SkEdge(write.node_id, wait.node_id, "memory", eid))

    return edges, nodes


# ============================================================================
# 事件颜色分配
# ============================================================================

EVENT_PALETTE = [
    "#E91E63",
    "#9C27B0",
    "#673AB7",
    "#3F51B5",
    "#00BCD4",
    "#009688",
    "#8BC34A",
    "#CDDC39",
    "#FFC107",
    "#FF5722",
    "#795548",
    "#607D8B",
    "#FF6F61",
    "#6B5B95",
    "#88B04B",
    "#92A8D1",
    "#034F84",
    "#F7CAC9",
    "#F7786B",
    "#CB99C9",
]


def assign_event_colors(all_rounds):
    """为所有 eventId 分配唯一颜色"""
    all_eids = set()
    for rd in all_rounds:
        for node in rd.all_nodes.values():
            if node.event_id:
                all_eids.add(node.event_id)
    colors = {}
    for i, eid in enumerate(sorted(all_eids)):
        colors[eid] = EVENT_PALETTE[i % len(EVENT_PALETTE)]
    return colors


# ============================================================================
# HTML 生成
# ============================================================================

# 节点颜色配置
NODE_COLORS = {
    "Kernel": {"bg": "#43A047", "border": "#2E7D32", "text": "#FFF"},
    "EventNotify": {"bg": "#FB8C00", "border": "#EF6C00", "text": "#FFF"},
    "EventWait": {"bg": "#1E88E5", "border": "#1565C0", "text": "#FFF"},
    "EventReset": {"bg": "#E53935", "border": "#C62828", "text": "#FFF"},
    "MemoryWrite": {"bg": "#AB47BC", "border": "#8E24AA", "text": "#FFF"},
    "MemoryWait": {"bg": "#26A69A", "border": "#00897B", "text": "#FFF"},
    "Default": {"bg": "#78909C", "border": "#546E7A", "text": "#FFF"},
    "Unknown": {"bg": "#757575", "border": "#616161", "text": "#FFF"},
}


def _js_obj(d):
    """Python dict -> JSON 字符串（用于嵌入 HTML）"""
    return json.dumps(d, ensure_ascii=False)


def generate_html(
    all_rounds, event_colors, log_files, output_path, init_nodes=None, mode="scope"
):
    """生成交互式 HTML 可视化文件

    mode: 仅支持 'scope'，用于区分视图数据来源（固定为 scope 库驱动）
    """

    # 准备每轮的 JSON 数据
    rounds_json_list = []
    for rd in all_rounds:
        edges, merged_nodes = build_edges(rd, init_nodes)
        rounds_json_list.append(
            {
                "nodes": [
                    merged_nodes[nid].to_dict() for nid in sorted(merged_nodes.keys())
                ],
                "custom_update_nodes": [
                    node.to_dict() for node in getattr(rd, "custom_update_nodes", [])
                ],
                "synthesized_custom_annotations": getattr(
                    rd, "synthesized_custom_annotations", []
                ),
                "scopes": [sc.to_dict() for sc in rd.scopes],
                "edges": [
                    {
                        "src": e.src_id,
                        "dst": e.dst_id,
                        "type": e.edge_type,
                        "event_id": e.event_id,
                    }
                    for e in edges
                ],
                "update_edges": [
                    {
                        "src": e.src_id,
                        "dst": e.dst_id,
                        "type": e.edge_type,
                        "event_id": e.event_id,
                    }
                    for e in getattr(rd, "update_edges", [])
                ],
                "update_summary": getattr(rd, "update_summary", {}),
                "source_file": rd.source_file,
                "source_round_idx": rd.source_round_idx,
                "model_instance_count": len(getattr(rd, "model_instances", []) or []),
            }
        )

    rounds_json = _js_obj(rounds_json_list)
    event_colors_json = _js_obj(event_colors)
    node_colors_json = _js_obj(NODE_COLORS)

    default_round = max(len(all_rounds) - 1, 0)
    header_html = render_page_header(
        icon="SC",
        kicker="",
        title="Scope View",
        note_html="当前页面按 ScopeName 查看单个 Scope 的节点排布、关系连边和 Update 信息。优先切换 ScopeName，再结合图和详情做定位。",
        stat_badges_html="",
    )
    output_dir = os.path.dirname(os.path.abspath(output_path))
    reports_dir = None
    cursor = output_dir
    while cursor and cursor != os.path.dirname(cursor):
        if os.path.basename(cursor) == "reports":
            reports_dir = cursor
            break
        cursor = os.path.dirname(cursor)
    portal_href = (
        os.path.relpath(os.path.join(reports_dir, "run-portal.html"), output_dir)
        if reports_dir
        else "../run-portal.html"
    )
    view_nav_html = render_view_nav(
        [
            ("导航页面", portal_href, False),
            ("Scope View", "scope-graph.html", True),
            ("TaskQue View", "task-queue-graph.html", False),
            ("Analysis View", "performance-report.html", False),
            ("DFX View", "hang-crash-report.html", False),
        ],
        kicker="",
    )
    legend_html = render_scope_section_block(
        [
            (
                "节点类型",
                [
                    make_scope_explainer_box(
                        "算子", color="#43A047", extra_class="structure-only"
                    ),
                    make_scope_explainer_box(
                        "事件通知", color="#FB8C00", extra_class="structure-only"
                    ),
                    make_scope_explainer_box(
                        "事件等待", color="#1E88E5", extra_class="structure-only"
                    ),
                    make_scope_explainer_box(
                        "事件重置", color="#E53935", extra_class="structure-only"
                    ),
                    make_scope_explainer_box(
                        "内存写入", color="#AB47BC", extra_class="structure-only"
                    ),
                    make_scope_explainer_box(
                        "内存等待", color="#26A69A", extra_class="structure-only"
                    ),
                    make_scope_explainer_box(
                        "SK", color="#43A047", extra_class="update-only", hidden=True
                    ),
                    make_scope_explainer_box(
                        "Value Write",
                        color="#8E24AA",
                        extra_class="update-only",
                        hidden=True,
                    ),
                    make_scope_explainer_box(
                        "Value Wait",
                        color="#00897B",
                        extra_class="update-only",
                        hidden=True,
                    ),
                    make_scope_explainer_box(
                        "Invalid",
                        color="#9ca3af",
                        extra_class="update-only",
                        hidden=True,
                    ),
                ],
            ),
            (
                "连边关系",
                [
                    make_scope_explainer_line(
                        "流内连边", color="#a3a3a3", extra_class="structure-only"
                    ),
                    make_scope_explainer_line(
                        "事件匹配边 (Notify→Wait)",
                        color="#E91E63",
                        dash_style="dashed",
                        extra_class="structure-only",
                    ),
                    make_scope_explainer_line(
                        "内存匹配边 (Write→Wait)",
                        color="#AB47BC",
                        dash_style="dashed",
                        extra_class="structure-only",
                    ),
                    make_scope_explainer_line(
                        "关系边",
                        color="#AB47BC",
                        dash_style="dashed",
                        extra_class="update-only",
                        hidden=True,
                    ),
                ],
            ),
            (
                "视图说明",
                [
                    make_scope_explainer_text(
                        "结构模式展示 Scope 内部的真实节点和依赖关系。",
                        extra_class="structure-only",
                    ),
                    make_scope_explainer_text(
                        "Update 模式聚焦 Value Write / Value Wait / Invalid。",
                        extra_class="update-only",
                        hidden=True,
                    ),
                ],
            ),
        ]
    )
    toolbar_html = render_graph_toolbar(
        label="ScopeName 切换",
        nav_html=render_graph_nav(
            "scopePrevBtn", "scopeNextBtn", "上一个 Scope", "下一个 Scope"
        ),
        select_html='<select class="toolbar-input graph-select" id="scopeSelect" style="width:560px; min-width:560px; max-width:560px;"></select>',
        index_chip_html=render_graph_index_chip("scopeIndexChip", "Scope 1 / 1"),
        trailing_html="""
        <input type="search" class="toolbar-input scope-search" id="scopeSearchInput" placeholder="查找 ScopeName、Scope ID、节点名" />
        <div class="mode-switch">
            <button type="button" class="mode-btn active" id="modeStructure">结构</button>
            <button type="button" class="mode-btn" id="modeUpdate">Update</button>
        </div>
        """,
        controls_html="""
        <button class="toolbar-btn" id="zoomIn">放大</button>
        <button class="toolbar-btn" id="zoomOut">缩小</button>
        <button class="toolbar-btn" id="zoomReset">重置</button>
        <button class="toolbar-btn" id="zoomFit">适配视图</button>
        <button class="toolbar-btn" id="kbdToggleBtn" onclick="toggleKeyboardNav()">键盘导航：开</button>
        <select id="kbdModeSelect" class="toolbar-btn" style="min-width:220px;">
            <option value="wasd_zoom">WASD：左右平移 + 缩放</option>
            <option value="wasd_pan">WASD：四向平移</option>
            <option value="arrows_zoom">方向键：左右平移 + 缩放</option>
            <option value="vim_zoom">H/L 平移 + K/J 缩放</option>
        </select>
        <label>缩放：</label>
        <span class="zoom-display" id="zoomLvl">100%</span>
        """,
    )
    scope_table_html = """
        <div style="overflow-x:auto;max-height:500px;overflow-y:auto;" class="detail-table-wrap">
            <table class="detail-table">
                <thead><tr id="tHead"></tr></thead>
                <tbody id="tBody"></tbody>
            </table>
        </div>
    """
    scope_pager_html = render_standard_table_pager_html(
        prev_btn_id="tablePrevBtn",
        next_btn_id="tableNextBtn",
        status_id="tablePageStatus",
    )
    scope_tools_html = render_standard_table_tools_html(
        main_html="""
        <input type="search" class="table-search" id="tableSearchInput" placeholder="查找节点、类型、地址、规则" />
        <select class="table-filter" id="tableTypeFilter"></select>
        """,
        jump_input_id="tableJumpInput",
        jump_btn_id="tableJumpBtn",
        row_count_id="tableRowCount",
        page_size_id="tablePageSize",
        page_size=20,
    )
    detail_panel_html = render_detail_table_panel(
        title="Scope 详情",
        title_id="tTitle",
        toggle_id="tableToggleBtn",
        content_id="tableContent",
        tools_html=scope_tools_html,
        table_html=scope_table_html,
        pager_html=scope_pager_html,
        initially_collapsed=True,
    )
    scope_styles = build_visualizer_styles(f"""
{COMMON_GRAPH_FRAME_CSS}
{COMMON_GRAPH_TOOLBAR_CSS}
{COMMON_SCOPE_SECTION_CSS}
{COMMON_TABLE_CONTROLS_CSS}
{COMMON_DETAIL_TABLE_CSS}
{COMMON_TOOLBAR_CSS}
{COMMON_TOOLTIP_CSS}
.page-shell {{ width: min(1760px, calc(100vw - 32px)); margin: 18px auto 28px; display:flex; flex-direction:column; gap:14px; position:relative; z-index:1; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:20px; }}
.card {{
    background:var(--sk-surface);
    border-radius:var(--sk-radius-md);
    padding:14px 16px;
    border:1px solid var(--sk-line);
    box-shadow:var(--sk-shadow-sm);
    border-left:4px solid var(--sk-accent);
}}
.card .lbl {{ font-size:11px; color:var(--sk-muted); text-transform:uppercase; letter-spacing:.4px; }}
.card .val {{ font-size:26px; font-weight:700; color:var(--sk-ink); margin-top:2px; }}
.mode-switch {{ display:flex; gap:8px; margin:0; }}
.mode-btn {{
    padding:8px 14px; border:1px solid var(--sk-line); border-radius:999px; background:#fff;
    color:#475569; font-size:13px; font-weight:600; cursor:pointer;
}}
.mode-btn.active {{ color:var(--sk-accent); border-color:#93c5fd; background:#eff6ff; }}
.controls {{ background: transparent; padding: 0; display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-size: var(--sk-text-md); }}
.zoom-display {{ color:#1d4ed8; font-weight:var(--sk-weight-bold); font-size:var(--sk-text-sm); min-width:42px; text-align:center; }}
.table-search {{ min-width:220px; }}
.scope-search {{ min-width:260px; }}
.badge {{
    display:inline-block; padding:2px 8px; border-radius:10px;
    font-size:11px; font-weight:600; color:#fff;
}}
.scope-tag {{
    display:inline-block; padding:1px 6px; border-radius:3px;
    font-size:11px; background:#e3f2fd; color:#1565c0;
}}
.kt-tag {{
    display:inline-block; padding:1px 6px; border-radius:3px;
    font-size:11px; font-weight:600; color:#fff;
}}
.kt-aic {{ background:#FF7043; }}
.kt-aiv {{ background:#42A5F5; }}
.kt-mix {{ background:#AB47BC; }}
.kt-default {{ background:#9E9E9E; }}
""")

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scope View</title>
<style>{scope_styles}</style>
</head>
<body>
<div class="page-shell">
    {header_html}
    {view_nav_html}
    {toolbar_html}
    <div class="summary" id="summary"></div>

    {legend_html}

    <div class="graph-frame">
        <h2 id="gTitle">Scope 关系图</h2>
        <div class="svg-container" id="svgWrap"></div>
    </div>

    {detail_panel_html}
</div>

<div class="tip" id="tip"></div>

<script>
// 嵌入数据
const ROUNDS = {rounds_json};
const EVC = {event_colors_json};
const NC = {node_colors_json};
// 布局参数
const NW = 156, NH = 40, LANE_H = 58, LANE_GAP = 12, HGAP = 184;
const MG = {{ left:210, right:60, top:36, bottom:40 }};
const VIS = {{
    node: {{
        radiusSm: {SVG_NODE_TOKENS["radius_sm"]},
        radiusMd: {SVG_NODE_TOKENS["radius_md"]},
        borderWidth: {SVG_NODE_TOKENS["border_width"]},
        borderWidthEmphasis: {SVG_NODE_TOKENS["border_width_emphasis"]},
        borderWidthActive: {SVG_NODE_TOKENS["border_width_active"]},
        titleFontSize: {SVG_NODE_TOKENS["title_font_size"]},
        subtitleFontSize: {SVG_NODE_TOKENS["subtitle_font_size"]},
        badgeFontSize: {SVG_NODE_TOKENS["badge_font_size"]},
        laneTitleFontSize: {SVG_NODE_TOKENS["lane_title_font_size"]},
    }},
    edge: {{
        widthStream: {SVG_EDGE_TOKENS["width_stream"]},
        widthMatch: {SVG_EDGE_TOKENS["width_match"]},
        widthEmphasis: {SVG_EDGE_TOKENS["width_emphasis"]},
        dashMemory: '{SVG_EDGE_TOKENS["dash_memory"]}',
        dashEvent: '{SVG_EDGE_TOKENS["dash_event"]}',
        labelRadiusSm: {SVG_EDGE_TOKENS["label_radius_sm"]},
        labelFontSize: {SVG_EDGE_TOKENS["label_font_size"]},
    }},
    lane: {{
        stripeRadius: {SVG_LANE_TOKENS["stripe_radius"]},
        lineWidth: {SVG_LANE_TOKENS["line_width"]},
    }},
}};

let curRound = {default_round};
let viewMode = 'structure';
let scopeSearchQuery = '';
let tableCollapsed = false;
let tableSearchQuery = '';
let tableTypeFilter = 'all';
let tablePage = 1;
let tablePageSize = 20;
let keyboardNavEnabled = true;
let keyboardNavMode = 'wasd_zoom';

// 缩放/平移状态（共享视口内核）
let scale = 1;
let translateX = 0;
let translateY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panStartTransX = 0;
let panStartTransY = 0;

function init() {{
    const preferredRound = findPreferredRoundIndex(getGraphContext());
    if (preferredRound >= 0) curRound = preferredRound;
    const structureBtn = document.getElementById('modeStructure');
    const updateBtn = document.getElementById('modeUpdate');
    const scopeSelect = document.getElementById('scopeSelect');
    const scopeSearchInput = document.getElementById('scopeSearchInput');
    const scopePrevBtn = document.getElementById('scopePrevBtn');
    const scopeNextBtn = document.getElementById('scopeNextBtn');
    const tableToggleBtn = document.getElementById('tableToggleBtn');
    const tableToggleHead = document.querySelector('#tableContent') ? document.querySelector('#tableContent').closest('.detail-panel')?.querySelector('[data-detail-toggle]') : null;
    const tableSearchInput = document.getElementById('tableSearchInput');
    const tableTypeSelect = document.getElementById('tableTypeFilter');
    const tablePrevBtn = document.getElementById('tablePrevBtn');
    const tableNextBtn = document.getElementById('tableNextBtn');
    const tablePageSizeSelect = document.getElementById('tablePageSize');
    const tableJumpInput = document.getElementById('tableJumpInput');
    const tableJumpBtn = document.getElementById('tableJumpBtn');
    const kbdModeSelect = document.getElementById('kbdModeSelect');
    if (structureBtn && updateBtn) {{
        structureBtn.onclick = () => setViewMode('structure');
        updateBtn.onclick = () => setViewMode('update');
    }}
    if (scopeSelect) {{
        scopeSelect.onchange = (ev) => {{
            const next = parseInt(ev.target.value || '0', 10);
            if (!Number.isNaN(next)) {{
                curRound = next;
                showRound(curRound);
            }}
        }};
    }}
    if (scopeSearchInput) {{
        scopeSearchInput.value = scopeSearchQuery;
        scopeSearchInput.oninput = (ev) => {{
            scopeSearchQuery = ev.target.value || '';
            renderScopePicker();
        }};
    }}
    if (scopePrevBtn) {{
        scopePrevBtn.onclick = () => moveScopeRound(-1);
    }}
    if (scopeNextBtn) {{
        scopeNextBtn.onclick = () => moveScopeRound(1);
    }}
    if (tableToggleBtn) {{
        tableToggleBtn.onclick = (ev) => {{
            ev.stopPropagation();
            toggleTableCollapsed();
        }};
    }}
    if (tableToggleHead) {{
        tableToggleHead.onclick = (ev) => {{
            const target = ev.target;
            if (target && target.closest('input, select, button, a, textarea')) return;
            toggleTableCollapsed();
        }};
        tableToggleHead.onkeydown = (ev) => {{
            if (ev.key !== 'Enter' && ev.key !== ' ') return;
            ev.preventDefault();
            toggleTableCollapsed();
        }};
    }}
    if (tableSearchInput) {{
        tableSearchInput.value = tableSearchQuery;
        tableSearchInput.oninput = (ev) => {{
            tableSearchQuery = ev.target.value || '';
            tablePage = 1;
            renderTable(ROUNDS[curRound]);
        }};
    }}
    if (tableTypeSelect) {{
        tableTypeSelect.onchange = (ev) => {{
            tableTypeFilter = ev.target.value || 'all';
            tablePage = 1;
            renderTable(ROUNDS[curRound]);
        }};
    }}
    if (tablePrevBtn) {{
        tablePrevBtn.onclick = () => {{
            if (tablePage > 1) {{
                tablePage -= 1;
                renderTable(ROUNDS[curRound]);
            }}
        }};
    }}
    if (tableNextBtn) {{
        tableNextBtn.onclick = () => {{
            tablePage += 1;
            renderTable(ROUNDS[curRound]);
        }};
    }}
    if (tablePageSizeSelect) {{
        tablePageSizeSelect.onchange = (ev) => {{
            tablePageSize = parseInt(ev.target.value || '20', 10) || 20;
            tablePage = 1;
            renderTable(ROUNDS[curRound]);
        }};
    }}
    const jumpTablePage = () => {{
        if (!tableJumpInput) return;
        const target = parseInt(tableJumpInput.value || '', 10);
        if (!Number.isFinite(target)) return;
        tablePage = target;
        renderTable(ROUNDS[curRound]);
    }};
    if (tableJumpBtn) {{
        tableJumpBtn.onclick = jumpTablePage;
    }}
    if (tableJumpInput) {{
        tableJumpInput.onkeydown = (ev) => {{
            if (ev.key === 'Enter') jumpTablePage();
        }};
    }}
    if (kbdModeSelect) {{
        kbdModeSelect.value = keyboardNavMode;
        kbdModeSelect.onchange = (ev) => {{
            keyboardNavMode = ev.target.value || 'wasd_zoom';
            syncKeyboardModeControl();
        }};
    }}
    refreshTableFilterOptions();
    window.__SK_VIEWPORT_CONFIG = {{
        containerId: 'svgWrap',
        svgId: 'scopeSvg',
        minScale: 0.34,
        maxScale: 2.2,
        topPad: 24,
        sidePad: 28,
        fitMode: 'max',
    }};
    renderScopePicker(); showRound(curRound);
    document.getElementById('zoomIn').onclick = () => zoomIn();
    document.getElementById('zoomOut').onclick = () => zoomOut();
    document.getElementById('zoomReset').onclick = () => resetView();
    document.getElementById('zoomFit').onclick = () => fitView();
    attachViewportInteractions();
    attachKeyboardNav({{
        previous: () => moveScopeRound(-1),
        next: () => moveScopeRound(1),
        panStep: 80,
    }});
    syncKeyboardModeControl();
    syncTableCollapsedState();
}}

function getRoundScopeName(rd, roundIndex) {{
    if (rd && Array.isArray(rd.scopes) && rd.scopes.length) {{
        const scope = rd.scopes[0] || {{}};
        const names = Array.isArray(scope.names) ? scope.names : [];
        if (names.length && String(names[0] || '').trim()) return String(names[0]);
        if (scope.id != null) return 'Scope ' + scope.id;
    }}
    return 'Scope ' + (roundIndex + 1);
}}

function getRoundScopeSearchText(rd, roundIndex) {{
    const scopeName = getRoundScopeName(rd, roundIndex);
    const scopeIds = (rd.scopes || []).map(s => s.id).join(' ');
    const nodeNames = (rd.nodes || []).slice(0, 24).map(n => String(n.kernel_name || '')).join(' ');
    return [scopeName, scopeIds, nodeNames].join(' ').toLowerCase();
}}

function truncateScopeOptionLabel(value, maxChars = 72) {{
    const text = String(value || '');
    if (text.length <= maxChars) return text;
    return text.slice(0, maxChars - 1) + '…';
}}

function getVisibleRoundIndexes() {{
    const q = (scopeSearchQuery || '').trim().toLowerCase();
    const indexes = [];
    ROUNDS.forEach((rd, idx) => {{
        if (!q || getRoundScopeSearchText(rd, idx).includes(q)) indexes.push(idx);
    }});
    return indexes;
}}

function renderScopePicker() {{
    const select = document.getElementById('scopeSelect');
    const chip = document.getElementById('scopeIndexChip');
    const prevBtn = document.getElementById('scopePrevBtn');
    const nextBtn = document.getElementById('scopeNextBtn');
    if (!select) return;
    const visible = getVisibleRoundIndexes();
    if (!visible.length) {{
        select.innerHTML = '<option value="">没有匹配的 ScopeName</option>';
        select.disabled = true;
        if (chip) chip.textContent = 'Scope 0 / 0';
        if (prevBtn) prevBtn.disabled = true;
        if (nextBtn) nextBtn.disabled = true;
        return;
    }}
    if (!visible.includes(curRound)) curRound = visible[0];
    select.disabled = false;
    select.innerHTML = visible.map(idx => {{
        const rd = ROUNDS[idx];
        const scopeName = getRoundScopeName(rd, idx);
        return '<option value="' + idx + '" title="' + esc(scopeName) + '">' + esc(truncateScopeOptionLabel(scopeName)) + '</option>';
    }}).join('');
    select.value = String(curRound);
    const pos = visible.indexOf(curRound);
    if (chip) chip.textContent = 'Scope ' + (pos + 1) + ' / ' + visible.length;
    if (prevBtn) prevBtn.disabled = pos <= 0;
    if (nextBtn) nextBtn.disabled = pos >= visible.length - 1;
}}

function moveScopeRound(delta) {{
    const visible = getVisibleRoundIndexes();
    if (!visible.length) return;
    const pos = Math.max(0, visible.indexOf(curRound));
    const nextPos = Math.max(0, Math.min(visible.length - 1, pos + delta));
    curRound = visible[nextPos];
    renderScopePicker();
    showRound(curRound);
}}

function syncTableCollapsedState() {{
    const box = document.getElementById('tableContent');
    const btn = document.getElementById('tableToggleBtn');
    const head = box ? box.closest('.detail-panel')?.querySelector('[data-detail-toggle]') : null;
    const panel = box ? box.closest('.detail-panel') : null;
    if (box) box.classList.toggle('is-collapsed', tableCollapsed);
    if (panel) panel.classList.toggle('is-open', !tableCollapsed);
    if (btn) {{
        btn.textContent = tableCollapsed ? '展开' : '收起';
    }}
    if (head) {{
        head.setAttribute('aria-expanded', tableCollapsed ? 'false' : 'true');
    }}
}}

function toggleTableCollapsed() {{
    tableCollapsed = !tableCollapsed;
    syncTableCollapsedState();
}}

function getCurrentTableTypeOptions() {{
    if (viewMode === 'update') {{
        return [
            ['all', '全部类型'],
            ['KERNEL', 'SK'],
            ['VALUE_WRITE', 'Value Write'],
            ['VALUE_WAIT', 'Value Wait'],
            ['INVALID', 'Invalid'],
            ['SYNTHESIZED_CUSTOM', 'Synthesized Custom'],
        ];
    }}
    return [
        ['all', '全部类型'],
        ['Kernel', 'Kernel'],
        ['EventNotify', 'EventNotify'],
        ['EventWait', 'EventWait'],
        ['EventReset', 'EventReset'],
        ['MemoryWrite', 'MemoryWrite'],
        ['MemoryWait', 'MemoryWait'],
    ];
}}

function refreshTableFilterOptions() {{
    const select = document.getElementById('tableTypeFilter');
    if (!select) return;
    const opts = getCurrentTableTypeOptions();
    select.innerHTML = opts.map(([value, label]) => '<option value="' + value + '">' + label + '</option>').join('');
    const valid = new Set(opts.map(([value]) => value));
    if (!valid.has(tableTypeFilter)) tableTypeFilter = 'all';
    select.value = tableTypeFilter;
}}

function matchesTableRow(n, rd) {{
    const typeKey = viewMode === 'update' ? (n.update_type || '') : (n.type || '');
    if (tableTypeFilter !== 'all' && typeKey !== tableTypeFilter) return false;
    const q = (tableSearchQuery || '').trim().toLowerCase();
    if (!q) return true;
    const parts = [
        String(n.id || ''),
        String(n.type || ''),
        String(n.update_type || ''),
        String(n.stream_id || ''),
        String(n.scope_id || ''),
        String(n.event_id || ''),
        String(n.kernel_name || ''),
        String(n.kernel_type || ''),
        String(n.update_addr || ''),
        String(n.update_value || ''),
        String(n.update_flag || ''),
        String(n.update_args || ''),
        String(n.update_op_info_ptr || ''),
        String(n.update_func_handle || ''),
    ];
    (n.inherited_write_facts || []).forEach(item => {{
        parts.push(String(item.addr || ''));
        (item.values || []).forEach(v => parts.push(String(v || '')));
        (item.rules || []).forEach(v => parts.push(String(v || '')));
    }});
    (n.inherited_wait_facts || []).forEach(item => {{
        parts.push(String(item.addr || ''));
        (item.values || []).forEach(v => parts.push(String(v || '')));
        (item.flags || []).forEach(v => parts.push(String(v || '')));
        (item.rules || []).forEach(v => parts.push(String(v || '')));
    }});
    (rd.update_edges || []).filter(e => e.src === n.id || e.dst === n.id).forEach(e => {{
        parts.push(String(e.type || ''));
        parts.push(String(e.src || ''));
        parts.push(String(e.dst || ''));
    }});
    return parts.join(' ').toLowerCase().includes(q);
}}

function getViewNodes(rd) {{
    return viewMode === 'update' ? [...rd.nodes, ...(rd.custom_update_nodes || [])] : [...rd.nodes];
}}

function getFilteredTableNodes(rd) {{
    return getViewNodes(rd)
        .sort((a,b) => a.stream_id - b.stream_id || a.node_idx_in_stream - b.node_idx_in_stream)
        .filter(n => matchesTableRow(n, rd));
}}

function syncTablePagerState(totalRows) {{
    const totalPages = Math.max(1, Math.ceil(totalRows / tablePageSize));
    if (tablePage > totalPages) tablePage = totalPages;
    if (tablePage < 1) tablePage = 1;
    const prevBtn = document.getElementById('tablePrevBtn');
    const nextBtn = document.getElementById('tableNextBtn');
    const status = document.getElementById('tablePageStatus');
    const count = document.getElementById('tableRowCount');
    const size = document.getElementById('tablePageSize');
    const jumpInput = document.getElementById('tableJumpInput');
    if (prevBtn) prevBtn.disabled = tablePage <= 1;
    if (nextBtn) nextBtn.disabled = tablePage >= totalPages;
    if (status) status.textContent = '第 ' + tablePage + ' / ' + totalPages + ' 页';
    if (count) count.textContent = '共 ' + totalRows + ' 条';
    if (size) size.value = String(tablePageSize);
    if (jumpInput) jumpInput.max = String(totalPages);
}}

function setViewMode(mode) {{
    viewMode = mode;
    const structureBtn = document.getElementById('modeStructure');
    const updateBtn = document.getElementById('modeUpdate');
    if (structureBtn) structureBtn.classList.toggle('active', mode === 'structure');
    if (updateBtn) updateBtn.classList.toggle('active', mode === 'update');
    document.querySelectorAll('.structure-only').forEach(el => {{
        el.style.display = mode === 'update' ? 'none' : '';
    }});
    document.querySelectorAll('.update-only').forEach(el => {{
        el.style.display = mode === 'update' ? '' : 'none';
    }});
    tablePage = 1;
    refreshTableFilterOptions();
    showRound(curRound);
}}

function getGraphContext() {{
    const params = new URLSearchParams(window.location.search);
    const scopeId = params.get('scopeId');
    const nodeId = params.get('nodeId');
    return {{
        scopeId: scopeId !== null && scopeId !== '' ? parseInt(scopeId, 10) : null,
        nodeId: nodeId !== null && nodeId !== '' ? parseInt(nodeId, 10) : null,
    }};
}}

function hasGraphContext(ctx) {{
    return ctx.scopeId != null || ctx.nodeId != null;
}}

function findPreferredRoundIndex(ctx) {{
    if (!hasGraphContext(ctx)) return curRound;
    if (ctx.nodeId != null) {{
        for (let i = 0; i < ROUNDS.length; i++) {{
            if (ROUNDS[i].nodes.some(n => n.id === ctx.nodeId)) return i;
        }}
    }}
    if (ctx.scopeId != null) {{
        for (let i = 0; i < ROUNDS.length; i++) {{
            if (ROUNDS[i].nodes.some(n => n.scope_id === ctx.scopeId)) return i;
        }}
    }}
    return curRound;
}}

function updateGraphFocusSummary(state, rd) {{
    return;
}}

function clearGraphFocus(wrap) {{
    wrap.querySelectorAll('.nd').forEach(g => {{
        g.classList.remove('focus-primary', 'focus-secondary', 'unfocused');
        g.style.opacity = '';
        const rect = g.querySelector('rect');
        if (rect) {{
            rect.setAttribute('stroke-width', String(VIS.node.borderWidth));
            rect.setAttribute('opacity', '1');
        }}
    }});
}}

function applyGraphContext(rd, pos, nmap, wrap) {{
    const ctx = getGraphContext();
    const state = {{
        query: ctx,
        roundIndex: curRound,
        hit: false,
        primaryNodeId: null,
        focusIds: [],
    }};
    if (!hasGraphContext(ctx)) return state;

    clearGraphFocus(wrap);
    const focusIds = new Set();
    let primaryNodeId = null;

    if (ctx.scopeId != null) {{
        rd.nodes.forEach(n => {{
            if (n.scope_id === ctx.scopeId) {{
                focusIds.add(n.id);
                if (primaryNodeId === null) primaryNodeId = n.id;
            }}
        }});
    }}
    if (ctx.nodeId != null && nmap[ctx.nodeId]) {{
        focusIds.add(ctx.nodeId);
        primaryNodeId = ctx.nodeId;
    }}
    if (!focusIds.size || primaryNodeId === null || !pos[primaryNodeId]) {{
        return state;
    }}
    state.hit = true;
    state.primaryNodeId = primaryNodeId;
    state.focusIds = Array.from(focusIds);

    wrap.querySelectorAll('.nd').forEach(g => {{
        const nid = parseInt(g.dataset.nid, 10);
        if (!focusIds.has(nid)) {{
            g.classList.add('unfocused');
            g.style.opacity = '0.22';
            return;
        }}
        const rect = g.querySelector('rect');
        if (!rect) return;
        if (nid === primaryNodeId) {{
            g.classList.add('focus-primary');
            rect.setAttribute('stroke', '#ef4444');
            rect.setAttribute('stroke-width', String(VIS.node.borderWidthActive));
        }} else {{
            g.classList.add('focus-secondary');
            rect.setAttribute('stroke', '#f59e0b');
            rect.setAttribute('stroke-width', String(VIS.node.borderWidthEmphasis));
            rect.setAttribute('opacity', '0.96');
        }}
    }});

    const target = pos[primaryNodeId];
    const viewW = wrap.clientWidth || 1280;
    const viewH = wrap.clientHeight || 720;
    scale = Math.max(scale, 1);
    translateX = viewW / 2 - (target.x + NW / 2) * scale;
    translateY = viewH / 2 - (target.y + NH / 2) * scale;
    updateTransform();
    return state;
}}
function onKey(e) {{
    // 仅在非输入框时响应
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === '+' || e.key === '=') {{ zoomIn(); e.preventDefault(); }}
    else if (e.key === '-') {{ zoomOut(); e.preventDefault(); }}
    else if (e.key === '0') {{ fitView(); e.preventDefault(); }}
    else if (e.key === 'f' || e.key === 'F') {{ fitView(); e.preventDefault(); }}
}}

/* ---- 摘要 ---- */
function renderSummary(rd) {{
    const summaryEl = document.getElementById('summary');
    if (!summaryEl) return;
    const viewNodes = getViewNodes(rd);
    const n = viewNodes.length;
    const streams = new Set(viewNodes.map(x=>x.stream_id));
    const kernels = viewNodes.filter(x=>x.type==='Kernel').length;
    const events = n - kernels;
    const evEdges = rd.edges.filter(x=>x.type==='event').length;
    const memEdges = rd.edges.filter(x=>x.type==='memory').length;
    const stEdges = rd.edges.filter(x=>x.type==='stream').length;
    let cards = [
        ['总节点数', n], ['Stream 数', streams.size], ['Scope 数', rd.scopes.length],
        ['Kernel', kernels], ['Event', events], ['事件边', evEdges], ['流内边', stEdges],
    ];
    if (viewMode === 'update') {{
        const us = rd.update_summary || {{}};
        cards = [
            ['总节点数', n],
            ['Update 节点', us.updated_node_count || 0],
            ['Synthesized', us.synthesized_custom_count || 0],
            ['内存匹配边', us.notify_wait_edge_count || 0],
            ['地址组', us.addr_count || 0],
        ];
    }}
    summaryEl.innerHTML = cards.map(([l,v]) => '<div class="card"><div class="lbl">' + l + '</div><div class="val">' + v + '</div></div>').join('');
}}

/* ---- 切换 Scope ---- */
function showRound(idx) {{
    const rd = ROUNDS[idx];
    document.getElementById('gTitle').textContent = 'Scope 关系图';
    document.getElementById('tTitle').textContent = 'Scope 详情';
    renderScopePicker();
    renderSummary(rd);
    renderGraph(rd);
    renderTable(rd);
}}

/* ---- SVG 图 ---- */
function esc(s) {{ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function renderGraph(rd) {{
    const wrap = document.getElementById('svgWrap');
    const renderNodes = getViewNodes(rd);
    if (!renderNodes.length) {{ wrap.innerHTML = '<p style="color:#999;text-align:center;padding:40px">该轮次无节点数据</p>'; return; }}

    // 重置视口状态
    scale = 1;
    translateX = 0;
    translateY = 0;

    // 按 stream 分组
    const sg = {{}};
    renderNodes.forEach(n => {{ (sg[n.stream_id] = sg[n.stream_id]||[]).push(n); }});
    Object.values(sg).forEach(g => g.sort((a,b) => a.node_idx_in_stream - b.node_idx_in_stream));

    const sids = Object.keys(sg).map(Number).sort((a,b) => a - b);
    let maxN = 0;
    sids.forEach(sid => {{ if (sg[sid].length > maxN) maxN = sg[sid].length; }});

    const W = MG.left + maxN * HGAP + MG.right + NW;
    const H = MG.top + sids.length * (LANE_H + LANE_GAP) + MG.bottom;

    // 存储 SVG 内容尺寸（供 zoomFit 使用）
    wrap._svgW = W; wrap._svgH = H;

    // 计算位置
    const pos = {{}};
    sids.forEach((sid, si) => {{
        const y = MG.top + si * (LANE_H + LANE_GAP);
        sg[sid].forEach((n, ni) => {{ pos[n.id] = {{ x: MG.left + ni * HGAP, y }}; }});
    }});

    // 构建 node_id -> node 映射
    const nmap = {{}};
    renderNodes.forEach(n => nmap[n.id] = n);

    let s = '<svg id="scopeSvg" xmlns="http://www.w3.org/2000/svg" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '">';

    // 箭头 marker（defs 不放在 zoomGroup 内，因为通过 ID 引用）
    s += '<defs>';
    s += '<marker id="aS" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#aaa"/></marker>';
    s += '<marker id="aE" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#E91E63"/></marker>';
    s += '<marker id="aM" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#AB47BC"/></marker>';
    s += '<filter id="ds"><feDropShadow dx="1" dy="2" stdDeviation="2" flood-opacity=".12"/></filter>';
    // 为每种事件颜色创建 marker
    Object.entries(EVC).forEach(([eid, clr]) => {{
        const mid = 'aE_' + eid.replace(/[^a-zA-Z0-9]/g,'_');
        s += '<marker id="' + mid + '" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="' + clr + '"/></marker>';
    }});
    s += '</defs>';

    // 所有可视内容放入 zoomGroup，支持缩放平移
    s += '<g id="zoomGroup" transform="translate(0,0) scale(1)">';

    // Stream 背景条
    sids.forEach((sid, si) => {{
        const y = MG.top + si * (LANE_H + LANE_GAP) - 6;
        const cy = MG.top + si*(LANE_H+LANE_GAP) + NH/2;
        const laneLabel = sid === -1 ? 'Custom AIC' : (sid === -2 ? 'Custom AIV' : 'Stream ' + sid);
        s += '<rect x="0" y="' + y + '" width="' + W + '" height="' + (LANE_H + 12) + '" fill="' + (si%2===0?'#f8f9fb':'#fff') + '" rx="' + VIS.lane.stripeRadius + '"/>';
        s += '<line x1="' + (MG.left - 12) + '" y1="' + cy + '" x2="' + (W - MG.right + 8) + '" y2="' + cy + '" stroke="#d6dbe4" stroke-width="' + VIS.lane.lineWidth + '" opacity="0.95"/>';
        s += '<text x="16" y="' + (MG.top + si*(LANE_H+LANE_GAP) + NH/2 + 5) + '" font-size="' + VIS.node.laneTitleFontSize + '" font-weight="600" fill="#888" font-family="{VISUAL_FONT_MONO_STACK}">' + laneLabel + '</text>';
    }});

    // 绘制边
    const edgesToRender = viewMode === 'update' ? (rd.update_edges || []) : rd.edges;
    edgesToRender.forEach(e => {{
        const p1 = pos[e.src], p2 = pos[e.dst];
        if (!p1 || !p2) return;

        if (e.type === 'stream') {{
            const x1 = p1.x + NW, y1 = p1.y + NH/2, x2 = p2.x, y2 = p2.y + NH/2;
            s += '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="#ccc" stroke-width="' + VIS.edge.widthStream + '" marker-end="url(#aS)"/>';
        }} else if (e.type === 'memory' || e.type === 'update_notify_wait' || e.type === 'update_sk_write' || e.type === 'update_sk_wait') {{
            // 内存匹配边: MemoryWrite -> MemoryWait (虚线紫色)
            const clr = EVC[e.event_id] || '#AB47BC';
            const cx1 = p1.x + NW/2, cy1 = p1.y + NH/2;
            const cx2 = p2.x + NW/2, cy2 = p2.y + NH/2;
            const dy = cy2 - cy1;
            let off = dy === 0 ? 90 : 0;
            let cpx = (cx1+cx2)/2 + (dy < 0 ? off : -off);
            let cp1y = cy1, cp2y = cy2;
            s += '<path d="M '+cx1+' '+cy1+' C '+cpx+' '+cp1y+', '+cpx+' '+cp2y+', '+cx2+' '+cy2+'" stroke="'+clr+'" stroke-width="' + VIS.edge.widthMatch + '" stroke-dasharray="' + VIS.edge.dashMemory + '" fill="none" marker-end="url(#aM)" opacity=".75"/>';
            const lx = cpx, ly = (cy1+cy2)/2;
            let label = 'Memory';
            if (e.type === 'memory' && e.event_id) label = 'Mem:' + (e.event_id||'').slice(-8);
            else if (e.type === 'update_sk_write') label = 'ValueWrite→SK';
            else if (e.type === 'update_sk_wait') label = 'SK→ValueWait';
            else if (e.type === 'update_notify_wait') label = '关系边';
            const tw = label.length * 6.2 + 14;
            s += '<rect x="'+(lx-tw/2)+'" y="'+(ly-9)+'" width="'+tw+'" height="18" rx="' + VIS.edge.labelRadiusSm + '" fill="'+clr+'" opacity=".9"/>';
            s += '<text x="'+lx+'" y="'+(ly+4)+'" font-size="' + VIS.edge.labelFontSize + '" fill="#fff" text-anchor="middle" font-family="{VISUAL_FONT_MONO_STACK}">'+esc(label)+'</text>';
        }} else {{
            // 事件匹配边: Notify -> Wait (虚线彩色)
            const clr = EVC[e.event_id] || '#E91E63';
            const mid = 'aE_' + (e.event_id||'').replace(/[^a-zA-Z0-9]/g,'_');
            const cx1 = p1.x + NW/2, cy1 = p1.y + NH/2;
            const cx2 = p2.x + NW/2, cy2 = p2.y + NH/2;
            const dx = cx2 - cx1, dy = cy2 - cy1;
            // 贝塞尔偏移
            let off = dy === 0 ? 90 : 0;
            let cpx = (cx1+cx2)/2 + (dy < 0 ? off : -off);
            let cp1y = cy1, cp2y = cy2;
            s += '<path d="M '+cx1+' '+cy1+' C '+cpx+' '+cp1y+', '+cpx+' '+cp2y+', '+cx2+' '+cy2+'" stroke="'+clr+'" stroke-width="' + VIS.edge.widthMatch + '" stroke-dasharray="' + VIS.edge.dashEvent + '" fill="none" marker-end="url(#'+mid+')" opacity=".75"/>';
            // 事件 ID 标签
            const lx = cpx, ly = (cy1+cy2)/2;
            const shortEid = (e.event_id||'').slice(-8);
            const tw = shortEid.length * 5.4 + 10;
            s += '<rect x="'+(lx-tw/2)+'" y="'+(ly-9)+'" width="'+tw+'" height="18" rx="' + VIS.edge.labelRadiusSm + '" fill="'+clr+'" opacity=".9"/>';
            s += '<text x="'+lx+'" y="'+(ly+4)+'" font-size="' + VIS.edge.labelFontSize + '" fill="#fff" text-anchor="middle" font-family="{VISUAL_FONT_MONO_STACK}">'+esc(shortEid)+'</text>';
        }}
        if (e.type === 'update_notify_wait') {{
            const clr = '#b45309';
            const cx1 = p1.x + NW/2, cy1 = p1.y + NH/2;
            const cx2 = p2.x + NW/2, cy2 = p2.y + NH/2;
            const dy = cy2 - cy1;
            const off = dy === 0 ? 72 : 0;
            const cpx = (cx1 + cx2) / 2 + (dy < 0 ? off : -off);
            const cp1y = cy1, cp2y = cy2;
            s += '<path d="M '+cx1+' '+cy1+' C '+cpx+' '+cp1y+', '+cpx+' '+cp2y+', '+cx2+' '+cy2+'" stroke="'+clr+'" stroke-width="' + VIS.edge.widthEmphasis + '" fill="none" opacity=".9"/>';
            const lx = cpx, ly = (cy1 + cy2) / 2;
            const label = 'N→W';
            const tw = 36;
            s += '<rect x="'+(lx-tw/2)+'" y="'+(ly-9)+'" width="'+tw+'" height="18" rx="' + VIS.edge.labelRadiusSm + '" fill="'+clr+'" opacity=".92"/>';
            s += '<text x="'+lx+'" y="'+(ly+4)+'" font-size="' + VIS.edge.labelFontSize + '" fill="#fff" text-anchor="middle" font-family="{VISUAL_FONT_MONO_STACK}">'+label+'</text>';
        }}
    }});

    // 绘制节点
    renderNodes.forEach(n => {{
        const p = pos[n.id];
        if (!p) return;
        let c = NC[n.type] || NC['Unknown'];
        let l1 = '', l2 = '';
        if (n.type === 'Kernel') {{
            let nm = (n.kernel_name||'').replace('static_kernel_','').replace(/_d\\d+.*$/,'');
            l1 = nm.length > 17 ? nm.slice(0,15)+'..' : nm;
            const kt = n.kernel_type || '';
            l2 = kt ? kt + ' B:' + n.num_blocks : 'ID:' + n.id;
        }} else {{
            l1 = n.type.replace('Event','');
            l2 = n.event_id ? n.event_id.slice(-8) : '';
        }}
        if (viewMode === 'update' && n.update_type) {{
            l2 = '';
            if (n.update_type === 'KERNEL') {{
                c = NC['Kernel'] || c;
                l1 = 'SK';
            }} else if (n.update_type === 'INVALID') {{
                c = {{bg:'#9ca3af', border:'#6b7280', text:'#fff'}};
                l1 = 'Invalid';
            }} else if (n.update_type === 'VALUE_WAIT') {{
                c = NC['MemoryWait'] || c;
                l1 = 'Value Wait';
            }} else if (n.update_type === 'VALUE_WRITE') {{
                c = NC['MemoryWrite'] || c;
                l1 = 'Value Write';
            }} else if (n.update_type === 'SYNTHESIZED_CUSTOM') {{
                c = NC[n.type] || NC['Unknown'];
                l1 = n.type || 'Custom';
                l2 = n.queue_name || '';
            }}
        }}
        s += '<g class="nd" data-nid="'+n.id+'" data-scope-id="'+(n.scope_id==null?'':n.scope_id)+'" data-x="'+p.x+'" data-y="'+p.y+'" style="cursor:pointer">';
        s += '<rect x="'+p.x+'" y="'+p.y+'" width="'+NW+'" height="'+NH+'" rx="' + VIS.node.radiusSm + '" fill="'+c.bg+'" stroke="'+c.border+'" stroke-width="' + VIS.node.borderWidth + '" filter="url(#ds)"/>';
        s += '<text x="'+(p.x+NW/2)+'" y="'+(p.y+16)+'" font-size="' + VIS.node.titleFontSize + '" fill="'+c.text+'" text-anchor="middle" font-weight="600">'+esc(l1)+'</text>';
        if (l2) s += '<text x="'+(p.x+NW/2)+'" y="'+(p.y+30)+'" font-size="' + VIS.node.subtitleFontSize + '" fill="'+c.text+'" text-anchor="middle" font-family="{VISUAL_FONT_MONO_STACK}" opacity=".9">'+esc(l2)+'</text>';
        if (viewMode === 'update' && n.update_type) {{
            let ubg = '#475569', ulbl = n.update_type;
            if (n.update_type === 'KERNEL') {{
                ubg = '#166534'; ulbl = 'SK';
            }} else if (n.update_type === 'VALUE_WAIT') {{
                ubg = '#00897B'; ulbl = 'VWt';
            }} else if (n.update_type === 'VALUE_WRITE') {{
                ubg = '#8E24AA'; ulbl = 'VWr';
            }} else if (n.update_type === 'INVALID') {{
                ubg = '#6b7280'; ulbl = 'I';
            }} else if (n.update_type === 'SYNTHESIZED_CUSTOM') {{
                ubg = '#b45309'; ulbl = 'C';
            }}
            const badgeW = ulbl === 'SK' ? 26 : (ulbl.length > 2 ? 30 : 18);
            const badgeX = p.x + NW - badgeW - 6;
            const badgeTextX = badgeX + badgeW / 2;
            s += '<rect x="'+badgeX+'" y="'+(p.y-8)+'" width="'+badgeW+'" height="18" rx="' + VIS.node.radiusMd + '" fill="'+ubg+'" stroke="#fff" stroke-width="' + VIS.node.borderWidth + '"/>';
            s += '<text x="'+badgeTextX+'" y="'+(p.y+4)+'" font-size="' + VIS.node.badgeFontSize + '" fill="#fff" text-anchor="middle" font-weight="700">'+ulbl+'</text>';
        }}
        s += '</g>';
    }});

    s += '</g>';  // close zoomGroup
    s += '</svg>';

    wrap.innerHTML = '';
    wrap.insertAdjacentHTML('beforeend', s);
    attachViewportInteractions();
    requestAnimationFrame(() => {{ resetView(); }});

    // 绑定悬浮事件
    wrap.querySelectorAll('.nd').forEach(g => {{
        const nid = parseInt(g.dataset.nid);
        g.addEventListener('mouseenter', e => showTip(e, nid, rd, nmap));
        g.addEventListener('mousemove', moveTip);
        g.addEventListener('mouseleave', hideTip);
    }});
    const focusState = applyGraphContext(rd, pos, nmap, wrap);
    updateGraphFocusSummary(focusState, rd);
}}

/* ---- 悬浮提示 ---- */
const tipEl = document.getElementById('tip');
function getKtClass(kt) {{
    if (!kt) return 'kt-default';
    if (kt === 'AIC_ONLY' || kt === 'MIX_AIC_1_0') return 'kt-aic';
    if (kt === 'AIV_ONLY' || kt === 'MIX_AIV_1_0') return 'kt-aiv';
    return 'kt-mix';
}}

function getWaitRuleText(flagValue) {{
    if (!flagValue) return '';
    const fv = String(flagValue).trim().toLowerCase();
    if (fv === '0x0' || fv === '0') return 'write >= wait';
    if (fv === '0x1' || fv === '1') return 'write == wait';
    if (fv === '0x2' || fv === '2') return '(write & wait) != 0';
    if (fv === '0x3' || fv === '3') return '(~(write | wait)) != 0';
    return String(flagValue);
}}

function showTip(ev, nid, rd, nmap) {{
    const n = nmap[nid]; if (!n) return;
    if (viewMode === 'update') {{
        const effectiveType = (() => {{
            if (n.update_type === 'KERNEL') return 'KERNEL';
            if (n.update_type === 'VALUE_WAIT') return 'VALUE_WAIT';
            if (n.update_type === 'VALUE_WRITE') return 'VALUE_WRITE';
            if (n.update_type === 'SYNTHESIZED_CUSTOM') return 'SYNTHESIZED_CUSTOM';
            return n.update_type || 'UNKNOWN';
        }})();
        let h = '<b>Node #' + n.id + '</b><br>';
        h += 'UpdateType: <b>' + esc(effectiveType) + '</b><br>';
        if (n.identity_kind) h += 'Identity: <span class="mono">' + esc(n.identity_kind) + '</span><br>';
        if (n.custom_instance_key) h += 'CustomKey: <span class="mono">' + esc(n.custom_instance_key) + '</span><br>';
        if (n.update_addr) h += 'Addr: <span class="mono">' + esc(n.update_addr) + '</span><br>';
        if (n.update_value) h += 'Value: <span class="mono">' + esc(n.update_value) + '</span><br>';
        if (n.update_flag) h += 'Flag: <span class="mono">' + esc(n.update_flag) + '</span><br>';
        if (n.update_type === 'VALUE_WAIT' && n.update_flag) h += 'Rule: <span class="mono">' + esc(getWaitRuleText(n.update_flag)) + '</span><br>';
        if (n.update_type === 'KERNEL') {{
            if (n.update_op_info_ptr) h += 'OpInfoPtr: <span class="mono">' + esc(n.update_op_info_ptr) + '</span><br>';
            if (n.update_op_info_size) h += 'OpInfoSize: <span class="mono">' + esc(String(n.update_op_info_size)) + '</span><br>';
            if (n.update_func_handle) h += 'FuncHandle: <span class="mono">' + esc(n.update_func_handle) + '</span><br>';
            if (n.update_args) h += 'Args: <span class="mono">' + esc(n.update_args) + '</span><br>';
            if (n.update_args_size) h += 'ArgsSize: <span class="mono">' + esc(String(n.update_args_size)) + '</span><br>';
            if (n.update_num_blocks) h += 'NumBlocks: <span class="mono">' + esc(String(n.update_num_blocks)) + '</span><br>';
        }} else if (n.update_type === 'SYNTHESIZED_CUSTOM') {{
            if (n.queue_name) h += 'Queue: <span class="mono">' + esc(n.queue_name) + '</span><br>';
            if (n.update_args) h += 'Args: <span class="mono">' + esc(n.update_args) + '</span><br>';
            if (n.raw_node_id != null) h += 'RawNodeId: <span class="mono">' + esc(String(n.raw_node_id)) + '</span><br>';
        }}
        const inheritedRows = [];
        (n.inherited_write_facts || []).forEach(item => {{
            let row = 'Value Write: addr [' + esc(item.addr || '') + ']';
            if ((item.values || []).length) row += ', value [' + esc((item.values || []).join(', ')) + ']';
            inheritedRows.push(row);
        }});
        (n.inherited_wait_facts || []).forEach(item => {{
            let row = 'Value Wait: addr [' + esc(item.addr || '') + ']';
            if ((item.values || []).length) row += ', value [' + esc((item.values || []).join(', ')) + ']';
            if ((item.rules || []).length) row += ', rule [' + esc((item.rules || []).join(', ')) + ']';
            inheritedRows.push(row);
        }});
        if (inheritedRows.length) {{
            h += '<br><b>SK 内部事件:</b><br>';
            inheritedRows.forEach(row => {{
                h += '&nbsp;&nbsp;' + esc(row) + '<br>';
            }});
        }}
        const conn = (rd.update_edges || []).filter(e => e.src === nid || e.dst === nid);
        if (conn.length) {{
            h += '<br><b>关系边:</b><br>';
            conn.forEach(e => {{
                const oid = e.src === nid ? e.dst : e.src;
                const dir = e.src === nid ? '&#x2192;' : '&#x2190;';
                let lbl = 'MemoryMatch';
                if (e.type === 'update_sk_write') lbl = 'ValueWrite→SK';
                else if (e.type === 'update_sk_wait') lbl = 'SK→ValueWait';
                h += '&nbsp;&nbsp;' + dir + ' Node #' + oid + ' [' + lbl + ']<br>';
            }});
        }}
        tipEl.innerHTML = h;
        tipEl.style.display = 'block';
        moveTip(ev);
        return;
    }}
    let h = '<b>Node #' + n.id + '</b> &nbsp;|&nbsp; Type: ' + n.type + '<br>';
    h += 'Stream: ' + n.stream_id + ' &nbsp;|&nbsp; StreamIdx: ' + n.stream_idx + ' &nbsp;|&nbsp; Pos: ' + n.node_idx_in_stream + '<br>';
    if (n.event_id) h += 'EventID: ' + n.event_id + ' &nbsp;|&nbsp; Flag: ' + n.event_flag + '<br>';
    if (n.type === 'Kernel' && n.kernel_name) {{
        h += '<br><b>KernelInfos:</b><br>';
        h += '&nbsp;&nbsp;funcName: <span class="mono">' + esc(n.kernel_name) + '</span><br>';
        if (n.kernel_type) h += '&nbsp;&nbsp;kernelType: <span class="kt-tag ' + getKtClass(n.kernel_type) + '">' + esc(n.kernel_type) + '</span><br>';
        if (n.task_ratio) h += '&nbsp;&nbsp;taskRatio: [' + n.task_ratio[0] + ', ' + n.task_ratio[1] + ']<br>';
        h += '&nbsp;&nbsp;numBlocks: ' + n.num_blocks + '<br>';
        h += '&nbsp;&nbsp;cubeNum: ' + n.cube_num + '<br>';
        h += '&nbsp;&nbsp;vecNum: ' + n.vec_num + '<br>';
        h += '&nbsp;&nbsp;resolvedNum: ' + n.resolved_num + '<br>';
    }} else if (n.kernel_name) {{
        h += 'Kernel: ' + n.kernel_name + '<br>';
    }}
    if (n.scope_id != null) h += '<br>Scope: ' + n.scope_id + '<br>';
    // 连边
    const conn = rd.edges.filter(e => e.src === nid || e.dst === nid);
    if (conn.length) {{
        h += '<br><b>Connections:</b><br>';
        conn.forEach(e => {{
            const oid = e.src === nid ? e.dst : e.src;
            const on = nmap[oid];
            const dir = e.src === nid ? '&#x2192;' : '&#x2190;';
            let lbl = 'Stream';
            if (e.type === 'event') lbl = 'Event(' + (e.event_id||'').slice(-8) + ')';
            else if (e.type === 'memory') lbl = 'Mem(' + (e.event_id||'').slice(-8) + ')';
            h += '&nbsp;&nbsp;' + dir + ' Node #' + oid + ' (' + (on ? on.type : '?') + ') [' + lbl + ']<br>';
        }});
    }}
    tipEl.innerHTML = h;
    tipEl.style.display = 'block';
    moveTip(ev);
}}
function hideTip() {{ tipEl.style.display = 'none'; }}
function moveTip(ev) {{
    let x = ev.clientX + 16, y = ev.clientY + 16;
    if (x + 460 > window.innerWidth) x = ev.clientX - 470;
    if (y + 200 > window.innerHeight) y = ev.clientY - 210;
    tipEl.style.left = x + 'px';
    tipEl.style.top = y + 'px';
}}

/* ---- 表格 ---- */
function renderTable(rd) {{
    refreshTableFilterOptions();
    const filtered = getFilteredTableNodes(rd);
    syncTablePagerState(filtered.length);
    const start = (tablePage - 1) * tablePageSize;
    const sorted = filtered.slice(start, start + tablePageSize);

    // 动态表头
    const thEl = document.getElementById('tHead');
    if (viewMode === 'update') {{
        thEl.innerHTML = '<th>Node ID</th><th>Update 类型</th><th>Addr</th><th>Value</th><th>Flag / Rule</th><th>SK Update</th><th>SK 内部事件</th><th>关系边</th>';
    }} else {{
        thEl.innerHTML = '<th>Node ID</th><th>类型</th><th>Stream ID</th><th>Event ID</th>'
            + '<th>Kernel 名称</th><th>Kernel Type</th><th>Blocks</th>'
            + '<th>Cube</th><th>Vec</th><th>Task Ratio</th><th>流内序号</th>';
    }}

    const tb = document.getElementById('tBody');
    if (viewMode === 'update') {{
        tb.innerHTML = sorted.map(n => {{
            const target = n.update_type || '-';
            const label = target === 'KERNEL' ? 'SK'
                : target === 'VALUE_WRITE' ? 'Value Write'
                : target === 'VALUE_WAIT' ? 'Value Wait'
                : target === 'INVALID' ? 'Invalid'
                : esc(target || '-');
            const colorMap = {{
                'KERNEL': '#166534',
                'VALUE_WRITE': '#8E24AA',
                'VALUE_WAIT': '#00897B',
                'INVALID': '#6b7280',
            }};
            const badgeBg = colorMap[target] || '#475569';
            const addr = n.update_addr ? '<span class="mono">' + esc(n.update_addr) + '</span>' : '-';
            const value = n.update_value ? '<span class="mono">' + esc(n.update_value) + '</span>' : '-';
            const flagRule = [];
            if (n.update_flag) flagRule.push('flag=' + esc(n.update_flag));
            if (target === 'VALUE_WAIT' && n.update_flag) flagRule.push('rule=' + esc(getWaitRuleText(n.update_flag)));
            const skUpdate = [];
            if (target === 'KERNEL') {{
                if (n.update_op_info_ptr) skUpdate.push('OpInfoPtr=' + esc(n.update_op_info_ptr));
                if (n.update_op_info_size) skUpdate.push('OpInfoSize=' + esc(String(n.update_op_info_size)));
                if (n.update_func_handle) skUpdate.push('FuncHandle=' + esc(n.update_func_handle));
                if (n.update_args) skUpdate.push('Args=' + esc(n.update_args));
                if (n.update_args_size) skUpdate.push('ArgsSize=' + esc(String(n.update_args_size)));
                if (n.update_num_blocks) skUpdate.push('NumBlocks=' + esc(String(n.update_num_blocks)));
            }}
            const inherited = [];
            (n.inherited_write_facts || []).forEach(item => {{
                let row = 'Value Write: ' + esc(item.addr || '');
                if ((item.values || []).length) row += ' / ' + esc((item.values || []).join(', '));
                inherited.push(row);
            }});
            (n.inherited_wait_facts || []).forEach(item => {{
                let row = 'Value Wait: ' + esc(item.addr || '');
                if ((item.values || []).length) row += ' / ' + esc((item.values || []).join(', '));
                if ((item.rules || []).length) row += ' / ' + esc((item.rules || []).join(', '));
                inherited.push(row);
            }});
            const conn = (rd.update_edges || []).filter(e => e.src === n.id || e.dst === n.id).map(e => {{
                const oid = e.src === n.id ? e.dst : e.src;
                if (e.type === 'update_sk_write') return 'ValueWrite→SK / Node ' + oid;
                if (e.type === 'update_sk_wait') return 'SK→ValueWait / Node ' + oid;
                return '关系边 / Node ' + oid;
            }});
            return '<tr>'
                + '<td style="font-weight:600">' + n.id + '</td>'
                + '<td><span class="badge" style="background:'+badgeBg+'">' + label + '</span></td>'
                + '<td>' + addr + '</td>'
                + '<td>' + value + '</td>'
                + '<td class="mono">' + (flagRule.length ? flagRule.join('<br>') : '-') + '</td>'
                + '<td class="mono" style="font-size:11px">' + (skUpdate.length ? skUpdate.join('<br>') : '-') + '</td>'
                + '<td class="mono" style="font-size:11px">' + (inherited.length ? inherited.join('<br>') : '-') + '</td>'
                + '<td class="mono" style="font-size:11px">' + (conn.length ? conn.join('<br>') : '-') + '</td>'
                + '</tr>';
        }}).join('');
        return;
    }}
    tb.innerHTML = sorted.map(n => {{
        const c = NC[n.type] || NC['Unknown'];
        const tname = n.type === 'Kernel' ? 'Kernel' : n.type;
        const kn = n.type === 'Kernel'
            ? '<span class="mono">' + esc((n.kernel_name||'').replace('static_kernel_','')) + '</span>'
            : '-';
        const eid = n.event_id ? '<span class="mono">' + esc(n.event_id) + '</span>' : '-';
        // Kernel info columns
        let ktHtml = '-', blkHtml = '-', cubeHtml = '-', vecHtml = '-', trHtml = '-';
        if (n.type === 'Kernel') {{
            if (n.kernel_type) ktHtml = '<span class="kt-tag ' + getKtClass(n.kernel_type) + '">' + esc(n.kernel_type) + '</span>';
            blkHtml = n.num_blocks;
            cubeHtml = n.cube_num;
            vecHtml = n.vec_num;
            trHtml = n.task_ratio ? n.task_ratio[0] + ':' + n.task_ratio[1] : '-';
        }}
        // 公共列
        let row = '<tr>'
            + '<td style="font-weight:600">' + n.id + '</td>'
            + '<td><span class="badge" style="background:'+c.bg+'">' + esc(tname) + '</span></td>'
            + '<td>' + n.stream_id + '</td>'
            + '<td>' + eid + '</td>'
            + '<td>' + kn + '</td>'
            + '<td>' + ktHtml + '</td>'
            + '<td style="text-align:center">' + blkHtml + '</td>'
            + '<td style="text-align:center">' + cubeHtml + '</td>'
            + '<td style="text-align:center">' + vecHtml + '</td>'
            + '<td style="text-align:center;font-family:{VISUAL_FONT_MONO_STACK};font-size:11px">' + trHtml + '</td>';
        row += '<td>' + n.node_idx_in_stream + '</td>';
        row += '</tr>';
        return row;
    }}).join('');
}}

{COMMON_SVG_VIEWPORT_JS}
{COMMON_KEYBOARD_NAV_JS}
init();
</script>
</body>
</html>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path


# ============================================================================
# Main
# ============================================================================


def _find_library_pair(base_dir):
    """Find a matching scope/graph library pair under a given root."""
    root = os.path.abspath(base_dir)
    if os.path.isfile(root):
        root = os.path.dirname(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"[ERROR] 输入路径不存在或不是目录: {base_dir}")

    def _bundle(root_dir):
        scope_path = os.path.join(root_dir, "scope-library.json")
        graph_path = os.path.join(root_dir, "graph-library.json")
        if os.path.isfile(scope_path) and os.path.isfile(graph_path):
            scope_mtime = os.path.getmtime(scope_path)
            graph_mtime = os.path.getmtime(graph_path)
            score = -(max(scope_mtime, graph_mtime))
            depth = len(os.path.relpath(root_dir, root).split(os.sep))
            return (depth, score, root_dir, scope_path, graph_path)
        return None

    candidates = []

    scan_roots = [root]
    for extra in ("reports", "reports/data", "data"):
        scan_roots.append(os.path.join(root, extra))

    for candidate_root in scan_roots:
        if os.path.isdir(candidate_root):
            if bundle := _bundle(candidate_root):
                candidates.append(bundle)

    for scan_root, _, _ in os.walk(root):
        bundle = _bundle(scan_root)
        if bundle:
            candidates.append(bundle)

    if not candidates:
        raise FileNotFoundError(
            f"[ERROR] 未在 {base_dir} 中找到 scope-library.json 与 graph-library.json 的成对库文件"
        )

    # 优先选择更浅层目录 + 更晚更新时间
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, bundle_root, scope_library, graph_library = candidates[0]
    return scope_library, graph_library, bundle_root


def main():
    parser = argparse.ArgumentParser(
        description="Scope 可视化 - 从 scope-library + graph-library 生成 Scope / Update 视图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 模式1：直接指定库文件\n"
            '  python sk_scope_visualizer.py --scope-library "$SCOPE_LIBRARY" --graph-library "$GRAPH_LIBRARY" -o scope-graph.html\n'
            "  # 模式2：给定目录，自动搜索匹配库对\n"
            '  python sk_scope_visualizer.py "$RUN_OR_RESULT_DIR" -o scope-graph.html'
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help="目录（自动查找 scope-library.json 和 graph-library.json），或直接指定目录路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="scope-graph.html",
        help="输出 HTML 文件路径（默认: scope-graph.html）",
    )
    parser.add_argument(
        "--scope-library",
        help="直接指定 scope-library.json 路径。和 --graph-library 必须同时提供。",
    )
    parser.add_argument(
        "--graph-library",
        help="直接指定 graph-library.json 路径。和 --scope-library 必须同时提供。",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        _emit(f"[ERROR] 路径不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    if (args.scope_library is None) ^ (args.graph_library is None):
        _emit(
            "Error: --scope-library 和 --graph-library 必须同时提供。", file=sys.stderr
        )
        sys.exit(1)

    if args.scope_library and args.graph_library:
        scope_library = args.scope_library
        graph_library = args.graph_library
        bundle_root = (
            os.path.dirname(scope_library) if os.path.isfile(scope_library) else "."
        )
    else:
        try:
            scope_library, graph_library, bundle_root = _find_library_pair(input_path)
        except FileNotFoundError as exc:
            _emit(exc, file=sys.stderr)
            sys.exit(1)
        _emit(f"[INFO] 自动匹配库文件: {bundle_root}")

    if not os.path.isfile(scope_library):
        _emit(f"[ERROR] scope 库不存在: {scope_library}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(graph_library):
        _emit(f"[ERROR] graph 库不存在: {graph_library}", file=sys.stderr)
        sys.exit(1)

    source = ScopeLibrarySource(scope_library, graph_library, mode="scope")
    model = ScopeGraphModel.from_libraries(source)
    if not model.rounds:
        _emit("[ERROR] scope-library 中未解析到可显示 scope", file=sys.stderr)
        sys.exit(1)

    ScopeGraphRenderer(args.output).render_html(model)
    scope_count = len(model.rounds)
    node_count = sum(len(rd.all_nodes) for rd in model.rounds)
    _emit(f"[OK] 可视化报告已生成: {args.output}")
    _emit(f"     scope count={scope_count}, node count={node_count}")


if __name__ == "__main__":
    main()
