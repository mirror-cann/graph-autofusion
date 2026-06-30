#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""SK Task Queue Visualizer

Generate task-queue 图 using scope/graph library artifacts (`scope-library.json`
and `graph-library.json`).

Usage:
    python sk_task_queue_visualizer.py --scope-library scope-library.json --graph-library graph-library.json
"""

import os
import sys
import json
import argparse
from typing import Dict, Optional

from sk_visualizer_shared import (
    COMMON_DETAIL_TABLE_CSS,
    COMMON_GRAPH_FRAME_CSS,
    COMMON_GRAPH_TOOLBAR_CSS,
    COMMON_SCOPE_SECTION_CSS,
    COMMON_VISUALIZER_CSS,
    COMMON_KEYBOARD_NAV_JS,
    COMMON_SVG_VIEWPORT_JS,
    COMMON_TABLE_CONTROLS_CSS,
    COMMON_TOOLBAR_CSS,
    COMMON_TOOLTIP_CSS,
    COMMON_EMPTY_STATE_CSS,
    SVG_EDGE_TOKENS,
    SVG_LANE_TOKENS,
    SVG_NODE_TOKENS,
    VISUAL_FONT_MONO_STACK,
    VISUAL_FONT_UI_STACK,
    make_scope_explainer_box,
    make_scope_explainer_line,
    make_scope_explainer_text,
    render_detail_table_panel,
    render_empty_note,
    render_graph_index_chip,
    render_graph_nav,
    render_graph_toolbar,
    render_standard_table_pager_html,
    render_standard_table_tools_html,
    render_scope_section_block,
    render_page_header,
    render_view_nav,
)


def _emit(message: object = "", *, file=None, end: str = "\n") -> None:
    stream = sys.stdout if file is None else file
    stream.write(f"{message}{end}")


# ======================== Data Classes ========================


class TaskQueueLibrarySource:
    """Resolve scope/graph library JSON input for task queue graph generation."""

    def __init__(self, scope_library_path: str, graph_library_path: str):
        self.scope_library_path = scope_library_path
        self.graph_library_path = graph_library_path


class TaskQueueModel:
    """Stable object wrapper around parsed task queue sections."""

    def __init__(self, result: dict):
        self.result = result

    @classmethod
    def from_libraries(cls, source: "TaskQueueLibrarySource") -> "TaskQueueModel":
        with open(source.scope_library_path, "r", encoding="utf-8") as handle:
            scope_library = json.load(handle)
        with open(source.graph_library_path, "r", encoding="utf-8") as handle:
            graph_library = json.load(handle)
        result = build_task_queue_result_from_scope_library(
            scope_library, graph_library
        )
        if not result.get("sections"):
            raise ValueError("No task queue sections were found in scope-library.")
        return cls(result)

    def to_dict(self) -> dict:
        return self.result


class TaskQueueRenderer:
    """Render the existing interactive task queue graph without rewriting its UI logic."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def render_html(self, model: "TaskQueueModel") -> None:
        generate_html(model.to_dict(), self.output_path)


def _scope_identity(section: dict, fused: Optional[dict], index: int) -> Dict[str, str]:
    scope_name = str(
        section.get("scope_name") or (fused or {}).get("scope_name") or ""
    ).strip()
    sk_id = str(section.get("sk_id") or (fused or {}).get("sk_id") or "").strip()
    start_node = str(
        section.get("start_node_name") or (fused or {}).get("start_node_name") or ""
    ).strip()
    end_node = str(
        section.get("end_node_name") or (fused or {}).get("end_node_name") or ""
    ).strip()
    normalized_scope = (
        scope_name if scope_name and scope_name not in {"(none)", "none", "-"} else ""
    )
    display_name = f"SK节点 #{index}"

    subtitle = f"SK {sk_id}" if sk_id else ""
    search_terms = " ".join(
        part
        for part in [
            display_name,
            subtitle,
            normalized_scope,
            start_node,
            end_node,
            sk_id,
            scope_name,
        ]
        if part
    )
    return {
        "display_name": display_name,
        "subtitle": subtitle,
        "search_terms": search_terms,
        "sk_id": sk_id,
        "raw_name": scope_name,
    }


def _task_resolved_graph_node_id(
    task: dict, ordinal_node_map: Dict[int, int]
) -> Optional[int]:
    if "graph_identity_valid" in task:
        if not task.get("graph_identity_valid"):
            return None
        resolved = task.get("resolved_graph_node_id")
        return int(resolved) if resolved not in (None, "") else None

    task_index = task.get("task_index")
    node_id = ordinal_node_map.get(task_index, task.get("node_id"))
    return int(node_id) if node_id not in (None, "") else None


def _library_task_to_payload(
    task: dict,
    queue_name: str,
    log_seq: int,
    section: dict,
    node_lookup: Dict[int, dict],
    ordinal_node_map: Dict[int, int],
) -> dict:
    dispatch_type = str(task.get("task_type", "") or "").upper()
    scope_name = section.get("scope_name", "")
    sync_type = str(task.get("sync_type", "") or "")
    node_id = _task_resolved_graph_node_id(task, ordinal_node_map)
    node_meta = node_lookup.get(int(node_id), {}) if node_id not in (None, "") else {}
    kernel_name = ""
    if dispatch_type in {"FUNC", "PRELOAD"}:
        kernel_name = str(node_meta.get("func_name") or scope_name or "")
    return {
        "task_id": task.get("task_id", log_seq),
        "task_idx": task.get("task_index", -1),
        "stask_idx": task.get("task_index", -1),
        "dispatch_type": dispatch_type,
        "queue": str(queue_name).lower(),
        "raw_log": task.get("detail", ""),
        "args": str(task.get("args", "") or ""),
        "entries": list(task.get("entries", []) or []),
        "kernel_queue_type": "",
        "kernel_name": kernel_name,
        "num_blocks": task.get("block_count", 0),
        "kernel_info": {},
        "sync_flag": sync_type,
        "prev_type": "",
        "next_type": "",
        "sync_type": sync_type,
        "sync_kind": str(task.get("sync_kind", "") or ""),
        "sync_direction": str(task.get("sync_direction", "") or ""),
        "sync_raw_args": str(task.get("sync_raw_args", task.get("args", "")) or ""),
        "sync_value": task.get("sync_value"),
        "event_id": "",
        "raw_node_id": task.get("raw_node_id", task.get("node_id")),
        "resolved_graph_node_id": node_id if node_id not in (None, "") else None,
        "graph_identity_valid": bool(task.get("graph_identity_valid"))
        if "graph_identity_valid" in task
        else node_id not in (None, ""),
        "graph_identity_source": str(task.get("graph_identity_source", ""))
        if task.get("graph_identity_source")
        else "",
        "graph_identity_reason": str(task.get("graph_identity_reason", ""))
        if task.get("graph_identity_reason")
        else "",
        "node_id": str(node_id) if node_id not in (None, "") else "",
        "node_type": str(node_meta.get("node_type") or ""),
        "log_seq": log_seq,
        "graph_node_id": int(node_id) if node_id not in (None, "") else -1,
        "stream_id": int(node_meta.get("stream_id", -1) or -1),
        "stream_idx_in_graph": int(node_meta.get("stream_idx_in_graph", -1) or -1),
    }


def _find_library_pair(base_dir: str) -> tuple[str, str, str]:
    """Find a matching scope/graph library pair under input directory."""
    root = os.path.abspath(base_dir)
    if os.path.isfile(root):
        root = os.path.dirname(root)

    if not os.path.isdir(root):
        raise FileNotFoundError(f"输入路径不存在或不是目录: {base_dir}")

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

    for walk_root, _, _ in os.walk(root):
        bundle = _bundle(walk_root)
        if bundle:
            candidates.append(bundle)

    if not candidates:
        raise FileNotFoundError(
            f"未在 {base_dir} 中找到 scope-library.json 与 graph-library.json"
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, bundle_root, scope_library, graph_library = candidates[0]
    return scope_library, graph_library, bundle_root


def build_task_queue_result_from_scope_library(
    scope_library: dict, graph_library: Optional[dict] = None
) -> dict:
    device_task_library = scope_library.get("device_task_library", {})
    fused_library = scope_library.get("fused_library", {})
    source_path = device_task_library.get("path", "")
    source_file = os.path.basename(source_path) if source_path else "scope-library.json"
    node_lookup = {
        int(node.get("node_id")): node
        for node in (graph_library or {}).get("node_library", {}).get("nodes", [])
        if node.get("node_id") is not None
    }
    fused_by_sk = {
        item.get("sk_id"): item
        for item in fused_library.get("functions", [])
        if item.get("sk_id") is not None
    }
    sections = []
    for index, section in enumerate(device_task_library.get("sections", []), start=1):
        fused = fused_by_sk.get(section.get("sk_id"))
        scope_identity = _scope_identity(section, fused, index)
        ordinal_node_map = {
            int(item.get("ordinal")): int(item.get("node_id"))
            for item in (fused or {}).get("node_details", [])
            if item.get("ordinal") is not None and item.get("node_id") is not None
        }
        aic_tasks = [
            _library_task_to_payload(
                task, "AIC", log_seq, section, node_lookup, ordinal_node_map
            )
            for log_seq, task in enumerate(section.get("queues", {}).get("AIC", []))
        ]
        aiv_tasks = [
            _library_task_to_payload(
                task,
                "AIV",
                len(aic_tasks) + log_seq,
                section,
                node_lookup,
                ordinal_node_map,
            )
            for log_seq, task in enumerate(section.get("queues", {}).get("AIV", []))
        ]
        sections.append(
            {
                "global_section_seq": index,
                "section_seq": index,
                "sk_label": scope_identity["display_name"],
                "sk_subtitle": scope_identity["subtitle"],
                "sk_search_terms": scope_identity["search_terms"],
                "sk_id": scope_identity["sk_id"],
                "sk_name": scope_identity["raw_name"],
                "source_file": source_file,
                "start_line": -1,
                "end_line": -1,
                "incomplete": False,
                "aic_count": len(aic_tasks),
                "aiv_count": len(aiv_tasks),
                "model_instances": [],
                "model_instance_count": 0,
                "aic_tasks": aic_tasks,
                "aiv_tasks": aiv_tasks,
            }
        )

    return {
        "sections": sections,
        "stats": {
            "files_scanned": 1 if sections else 0,
            "sections_found": len(sections),
            "sections_deduped": len(sections),
            "dedupe_mode": "library",
        },
    }


# ======================== HTML Generation ========================

# Color scheme
COLORS = {
    "FUNC": {"bg": "#a8d4b4", "border": "#96c8a4", "text": "#2d5a3d"},
    "PRELOAD": {"bg": "#a0c8e0", "border": "#90bcd4", "text": "#2d4a5e"},
    "SYNC": {"bg": "#e0c8a0", "border": "#d4bc94", "text": "#5a4a2d"},
    "EVENT_NOTIFY": {"bg": "#e0a8b8", "border": "#d49cac", "text": "#5a2d3d"},
    "EVENT_WAIT": {"bg": "#c0a8d8", "border": "#b49ccc", "text": "#3d2d52"},
    "EVENT_RESET": {"bg": "#b0c4cc", "border": "#a4b8c0", "text": "#2d3a40"},
}

TASK_TYPE_LABELS = {
    "FUNC": "Func",
    "PRELOAD": "Preload",
    "SYNC": "Sync",
    "EVENT_NOTIFY": "Notify",
    "EVENT_WAIT": "Wait",
    "EVENT_RESET": "Reset",
}


def generate_html(result: dict, output_path: str):
    """Generate the interactive HTML visualization."""
    sections = result.get("sections", [])
    stats = result.get("stats", {})
    output_file = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_file)

    sections_json = []
    for sec in sections:
        model_instance_list = sec.get("model_instances", [])
        sections_json.append(
            {
                "global_section_seq": sec.get("global_section_seq", 0),
                "section_seq": sec.get("section_seq", 0),
                "sk_label": sec.get("sk_label", ""),
                "sk_subtitle": sec.get("sk_subtitle", ""),
                "sk_search_terms": sec.get("sk_search_terms", ""),
                "sk_id": sec.get("sk_id", ""),
                "sk_name": sec.get("sk_name", ""),
                "source_file": sec.get("source_file", ""),
                "start_line": sec.get("start_line", -1),
                "end_line": sec.get("end_line", -1),
                "incomplete": bool(sec.get("incomplete", False)),
                "aic_count": sec.get("aic_count", 0),
                "aiv_count": sec.get("aiv_count", 0),
                "model_instances": model_instance_list,
                "model_instance_count": sec.get(
                    "model_instance_count", len(model_instance_list)
                ),
                "aic_tasks": list(sec.get("aic_tasks", [])),
                "aiv_tasks": list(sec.get("aiv_tasks", [])),
            }
        )

    if sections_json:
        current_idx = len(sections_json) - 1
        current_section = sections_json[current_idx]
    else:
        current_idx = 0
        current_section = {
            "sk_label": "SK节点 #0",
            "sk_subtitle": "",
            "sk_search_terms": "",
            "sk_id": "",
            "sk_name": "",
            "source_file": "",
            "start_line": -1,
            "end_line": -1,
            "incomplete": False,
            "aic_count": 0,
            "aiv_count": 0,
            "aic_tasks": [],
            "aiv_tasks": [],
        }

    sections_data_json = json.dumps(sections_json, ensure_ascii=False)
    aic_data_json = json.dumps(current_section["aic_tasks"], ensure_ascii=False)
    aiv_data_json = json.dumps(current_section["aiv_tasks"], ensure_ascii=False)
    section_count = len(sections_json)
    files_scanned = int(stats.get("files_scanned", 0))
    sections_found = int(stats.get("sections_found", section_count))
    sections_deduped = int(stats.get("sections_deduped", section_count))
    total_tasks = int(current_section["aic_count"]) + int(current_section["aiv_count"])
    header_html = render_page_header(
        icon="TQ",
        kicker="",
        title="TaskQue View",
        note_html="当前页面直接承接单个 SK节点的 device task。图中保留英文任务语义，优先看谁在发、谁在收、谁在等。",
        stat_badges_html="",
    )
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
            ("Scope View", "scope-graph.html", False),
            ("TaskQue View", "task-queue-graph.html", True),
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
                    make_scope_explainer_box("Func", color="#16a34a"),
                    make_scope_explainer_box("Preload", color="#2563eb"),
                    make_scope_explainer_box("Sync", color="#d97706"),
                    make_scope_explainer_box("EventNotify", color="#be185d"),
                    make_scope_explainer_box("EventWait", color="#6d28d9"),
                    make_scope_explainer_box("EventReset", color="#475569"),
                ],
            ),
            (
                "连边关系",
                [
                    make_scope_explainer_line("队列内顺序边", color="#cbd5e1"),
                    make_scope_explainer_line(
                        "同步匹配边（SET → WAIT）", color="#94a3b8", dash_style="dashed"
                    ),
                    make_scope_explainer_line("ALL-SYNC 边", color="#9bbdaa"),
                ],
            ),
            (
                "语义转换",
                [
                    make_scope_explainer_text("Notify → Value Write"),
                    make_scope_explainer_text("Wait → Value Wait"),
                    make_scope_explainer_text("Reset → Value Write"),
                ],
            ),
        ]
    )
    detail_empty_html = render_empty_note(
        "当前 SK节点没有可展示的任务明细。",
        dom_id="detailEmpty",
        hidden=True,
    )
    toolbar_html = render_graph_toolbar(
        label="SK节点切换",
        nav_html=render_graph_nav(
            "sectionPrevBtn", "sectionNextBtn", "上一个 SK节点", "下一个 SK节点"
        ),
        search_html='<input type="search" class="section-search toolbar-input" id="sectionSearchInput" placeholder="查找 SK节点名称、SK、起止节点名" />',
        select_html='<select id="sectionSelect" class="toolbar-input graph-select" style="min-width: 360px; max-width: 580px;"></select>',
        index_chip_html=render_graph_index_chip(
            "sectionIndexChip", f"SK节点 1 / {section_count}"
        ),
        trailing_html=f'<span class="graph-meta-note" id="sectionHint">[{section_count} 个 SK节点]</span>',
        controls_html="""
        <button class="ctrl-btn toolbar-btn" onclick="zoomIn()">&#x2795; 放大</button>
        <button class="ctrl-btn toolbar-btn" onclick="zoomOut()">&#x2796; 缩小</button>
        <button class="ctrl-btn toolbar-btn" onclick="resetView()">重置</button>
        <button class="ctrl-btn toolbar-btn" onclick="fitView()">适配视图</button>
        <button class="ctrl-btn toolbar-btn" id="kbdToggleBtn" onclick="toggleKeyboardNav()">键盘导航：开</button>
        <select id="kbdModeSelect" class="ctrl-btn toolbar-btn" style="min-width: 220px;">
            <option value="wasd_zoom">WASD：左右平移 + 缩放</option>
            <option value="wasd_pan">WASD：四向平移</option>
            <option value="arrows_zoom">方向键：左右平移 + 缩放</option>
            <option value="vim_zoom">H/L 平移 + K/J 缩放</option>
        </select>
        <label>缩放：</label>
        <span class="zoom-display" id="zoomLevel">100%</span>
        <div class="ctrl-sep"></div>
        <span class="filter-label">筛选</span>
        <div class="filter-group">
            <button class="filter-btn active" data-type="FUNC" onclick="toggleFilter(this)">Func</button>
            <button class="filter-btn active" data-type="PRELOAD" onclick="toggleFilter(this)">Preload</button>
            <button class="filter-btn active" data-type="SYNC" onclick="toggleFilter(this)">Sync</button>
            <button class="filter-btn active" data-type="EVENT" onclick="toggleFilter(this)">Event</button>
        </div>
        """,
    )
    summary_html = """
    <div class="summary-grid">
        <article class="summary-metric-card">
            <div class="summary-metric-label">AIC Queue</div>
            <div class="summary-metric-value" id="statAic">{aic_count}</div>
            <div class="summary-metric-note">当前 SK节点的 AIC 任务数</div>
        </article>
        <article class="summary-metric-card">
            <div class="summary-metric-label">AIV Queue</div>
            <div class="summary-metric-value" id="statAiv">{aiv_count}</div>
            <div class="summary-metric-note">当前 SK节点的 AIV 任务数</div>
        </article>
        <article class="summary-metric-card">
            <div class="summary-metric-label">Task Total</div>
            <div class="summary-metric-value" id="statTotal">{total_tasks}</div>
            <div class="summary-metric-note">当前 SK节点的任务总数</div>
        </article>
    </div>
    """.format(
        aic_count=current_section["aic_count"],
        aiv_count=current_section["aiv_count"],
        total_tasks=total_tasks,
    )
    detail_table_html = """
    <div class="detail-table-wrap">
        <table class="detail-table">
            <thead>
                <tr>
                    <th>Queue</th>
                    <th>Type</th>
                    <th>TaskIdx</th>
                    <th>Sync</th>
                    <th>Args</th>
                    <th>Entries</th>
                    <th>Func</th>
                </tr>
            </thead>
            <tbody id="detailTableBody"></tbody>
        </table>
    </div>
    """
    detail_pager_html = render_standard_table_pager_html(
        prev_btn_id="detailPrevBtn",
        next_btn_id="detailNextBtn",
        status_id="detailPageStatus",
    )
    detail_tools_html = render_standard_table_tools_html(
        main_html="""
        <input type="search" class="detail-search" id="detailSearchInput" placeholder="查找队列、类型、Args、Func" />
        <select class="detail-filter" id="detailQueueFilter">
            <option value="all">全部队列</option>
            <option value="AIC">AIC</option>
            <option value="AIV">AIV</option>
        </select>
        <select class="detail-filter" id="detailTypeFilter">
            <option value="all">全部类型</option>
            <option value="FUNC">Func</option>
            <option value="PRELOAD">Preload</option>
            <option value="SYNC">Sync</option>
            <option value="EVENT_NOTIFY">Notify</option>
            <option value="EVENT_WAIT">Wait</option>
            <option value="EVENT_RESET">Reset</option>
        </select>
        """,
        jump_input_id="detailJumpInput",
        jump_btn_id="detailJumpBtn",
        row_count_id="detailRowCount",
        page_size_id="detailPageSize",
        page_size=20,
    )
    detail_panel_html = render_detail_table_panel(
        title="任务详情",
        subtitle=(current_section.get("sk_name") or "当前 SK节点的任务参数和同步语义")
        + (
            " · 当前 SK节点的任务参数和同步语义"
            if current_section.get("sk_name")
            else ""
        ),
        subtitle_id="detailSubtitle",
        toggle_id="detailToggleBtn",
        content_id="detailContent",
        tools_html=detail_tools_html,
        table_html=detail_table_html,
        pager_html=detail_pager_html,
        empty_html=detail_empty_html,
        initially_collapsed=True,
    )
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TaskQue View</title>
<style>
{COMMON_VISUALIZER_CSS}
{COMMON_GRAPH_FRAME_CSS}
{COMMON_GRAPH_TOOLBAR_CSS}
{COMMON_SCOPE_SECTION_CSS}
{COMMON_TOOLBAR_CSS}
{COMMON_TABLE_CONTROLS_CSS}
{COMMON_DETAIL_TABLE_CSS}
{COMMON_TOOLTIP_CSS}
{COMMON_EMPTY_STATE_CSS}

/* ===== Controls ===== */
.section-search {{
    min-width: 220px;
}}
.controls {{
    background: transparent;
    padding: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: var(--sk-text-md);
}}
.ctrl-sep {{
    width: 1px;
    height: 22px;
    background: rgba(0,0,0,0.08);
    margin: 0 8px;
}}
.controls label {{ color: #8892a4; font-size: var(--sk-text-sm); }}
.zoom-display {{
    color: #1d4ed8;
    font-weight: var(--sk-weight-bold);
    font-size: var(--sk-text-sm);
    min-width: 42px;
    text-align: center;
}}

.filter-group {{
    display: flex;
    gap: 6px;
    margin-left: 4px;
}}
.filter-label {{ color: #8892a4; font-size: var(--sk-text-xs); font-weight: var(--sk-weight-medium); text-transform: uppercase; letter-spacing: 1px; margin-left: 4px; }}
.filter-btn {{
    padding: 6px 12px;
    border-radius: 999px;
    border: 1.5px solid rgba(0,0,0,0.08);
    background: #ffffff;
    color: #8892a4;
    cursor: pointer;
    font-size: var(--sk-text-xs);
    font-weight: var(--sk-weight-medium);
    transition: all 0.15s ease;
}}
.filter-btn:hover {{ border-color: rgba(0,0,0,0.16); color: #4a5568; }}
.filter-btn.active {{ color: #fff; }}
.filter-btn[data-type="FUNC"]        {{ border-color: rgba(76,175,80,0.25); }}
.filter-btn[data-type="FUNC"].active  {{ background: #4CAF50; border-color: #4CAF50; color: #fff; }}
.filter-btn[data-type="PRELOAD"]        {{ border-color: rgba(33,150,243,0.25); }}
.filter-btn[data-type="PRELOAD"].active {{ background: #2196F3; border-color: #2196F3; color: #fff; }}
.filter-btn[data-type="SYNC"]           {{ border-color: rgba(255,152,0,0.25); }}
.filter-btn[data-type="SYNC"].active    {{ background: #FF9800; border-color: #FF9800; color: #fff; }}
.filter-btn[data-type="EVENT"]          {{ border-color: rgba(233,30,99,0.25); }}
.filter-btn[data-type="EVENT"].active   {{ background: #E91E63; border-color: #E91E63; color: #fff; }}
.detail-search {{ min-width: 220px; }}
.mono {{
    font-family: var(--sk-font-mono);
    font-size: var(--sk-text-xs);
    word-break: break-all;
}}
.detail-snippet {{
    max-width: 360px;
    white-space: normal;
}}
/* ===== Animations ===== */
@keyframes dashFlow {{
    to {{ stroke-dashoffset: -18; }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@media (max-width: 1100px) {{
    .toolbar-row {{ align-items: flex-start; }}
}}
</style>
</head>
<body>
<div class="page-shell">
{header_html}
{view_nav_html}
{toolbar_html}
{summary_html}

<div class="tooltip" id="tooltip"></div>

{legend_html}
<div class="graph-frame">
    <h2 id="graphTitle">Task Queue 图</h2>
    <div class="svg-container" id="svgContainer">
        <svg id="mainSvg" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
</div>
{detail_panel_html}
</div>

<script>
// ==================== Data ====================
const sectionsData = {sections_data_json};
let currentSectionIndex = {current_idx};
const parseStats = {{
    filesScanned: {files_scanned},
    sectionsFound: {sections_found},
    sectionsDeduped: {sections_deduped},
}};

// Backward-compatible aliases for current section view (updated on section switch)
let aicData = {aic_data_json};
let aivData = {aiv_data_json};

// ==================== Config ====================
const CFG = {{
    kernelMinW: 150,
    kernelMaxW: 228,
    preloadMinW: 118,
    preloadMaxW: 172,
    compactMinW: 94,
    compactMaxW: 152,
    nodeH: 58,
    gapX: 16,
    gapY: 84,
    padTop: 42,
    padLeft: 140,
    streamLabelW: 110,
    streamGap: 172,
    arrowSize: 6,
    edgeEndpointPad: 10,
}};

const VIS = {{
    node: {{
        radiusMd: {SVG_NODE_TOKENS["radius_md"]},
        radiusLg: {SVG_NODE_TOKENS["radius_lg"]},
        radiusPill: {SVG_NODE_TOKENS["radius_pill"]},
        borderWidth: {SVG_NODE_TOKENS["border_width_emphasis"]},
        titleFontSize: {SVG_NODE_TOKENS["title_font_size"]},
        subtitleFontSize: {SVG_NODE_TOKENS["subtitle_font_size"]},
        badgeFontSize: {SVG_NODE_TOKENS["badge_font_size"]},
        badgeMonoFontSize: {SVG_NODE_TOKENS["badge_mono_font_size"]},
        queueLabelFontSize: {SVG_NODE_TOKENS["queue_label_font_size"]},
    }},
    edge: {{
        widthQueue: {SVG_EDGE_TOKENS["width_queue"]},
        widthSync: {SVG_EDGE_TOKENS["width_sync"]},
        widthAllSync: {SVG_EDGE_TOKENS["width_all_sync"]},
        dashSync: '{SVG_EDGE_TOKENS["dash_sync"]}',
        dashWait: '{SVG_EDGE_TOKENS["dash_wait"]}',
        dashSet: '{SVG_EDGE_TOKENS["dash_set"]}',
        labelRadiusSm: {SVG_EDGE_TOKENS["label_radius_sm"]},
        labelRadiusMd: {SVG_EDGE_TOKENS["label_radius_md"]},
        labelFontSize: {SVG_EDGE_TOKENS["label_font_size"]},
        labelFontSizeSm: {SVG_EDGE_TOKENS["label_font_size_sm"]},
    }},
    lane: {{
        bgRadius: {SVG_LANE_TOKENS["queue_bg_radius"]},
        labelRadius: {SVG_LANE_TOKENS["queue_label_radius"]},
        lineWidth: {SVG_LANE_TOKENS["line_width"]},
    }},
}};

// ==================== State ====================
let scale = 1.0;
let translateX = 0;
let translateY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let currentSlotW = 188;
let panStartTransX = 0;
let panStartTransY = 0;
let keyboardNavEnabled = true;
let keyboardNavMode = 'wasd_zoom';
let detailCollapsed = false;
let detailSearchQuery = '';
let detailQueueFilter = 'all';
let detailTypeFilter = 'all';
let detailPage = 1;
let detailPageSize = 20;
let sectionSearchQuery = '';
let sectionOptionIndexMap = [];

const activeFilters = new Set(['FUNC', 'PRELOAD', 'SYNC', 'EVENT']);

function getCurrentSection() {{
    if (!sectionsData.length) return null;
    if (currentSectionIndex < 0) currentSectionIndex = 0;
    if (currentSectionIndex >= sectionsData.length) currentSectionIndex = sectionsData.length - 1;
    return sectionsData[currentSectionIndex];
}}

function getTaskContext() {{
    const params = new URLSearchParams(window.location.search);
    const taskIndex = params.get('taskIndex');
    const nodeId = params.get('nodeId');
    return {{
        taskIndex: taskIndex !== null && taskIndex !== '' ? parseInt(taskIndex, 10) : null,
        nodeId: nodeId !== null && nodeId !== '' ? parseInt(nodeId, 10) : null,
    }};
}}

function hasTaskContext(ctx) {{
    return ctx.taskIndex != null || ctx.nodeId != null;
}}

function sectionMatchesContext(sec, ctx) {{
    const allTasks = [...(sec.aic_tasks || []), ...(sec.aiv_tasks || [])];
    return allTasks.some(task => {{
        const taskMatch = ctx.taskIndex != null && taskIndexOf(task) === ctx.taskIndex;
        const nodeMatch = ctx.nodeId != null && (
            task.graph_node_id === ctx.nodeId || task.node_id === ctx.nodeId
        );
        return taskMatch || nodeMatch;
    }});
}}

function findPreferredSectionIndex(ctx) {{
    if (!hasTaskContext(ctx) || !sectionsData.length) return currentSectionIndex;
    for (let i = 0; i < sectionsData.length; i++) {{
        if (sectionMatchesContext(sectionsData[i], ctx)) return i;
    }}
    return currentSectionIndex;
}}

function refreshCurrentSectionData() {{
    const sec = getCurrentSection();
    if (!sec) {{
        aicData = [];
        aivData = [];
        return;
    }}
    aicData = sec.aic_tasks || [];
    aivData = sec.aiv_tasks || [];
}}

function refreshHeaderStats() {{
    const sec = getCurrentSection();
    const aic = sec ? (sec.aic_count || 0) : 0;
    const aiv = sec ? (sec.aiv_count || 0) : 0;
    const total = aic + aiv;
    document.getElementById('statAic').textContent = String(aic);
    document.getElementById('statAiv').textContent = String(aiv);
    document.getElementById('statTotal').textContent = String(total);
}}

function refreshSectionHint() {{
    const hint = document.getElementById('sectionHint');
    const chip = document.getElementById('sectionIndexChip');
    const prevBtn = document.getElementById('sectionPrevBtn');
    const nextBtn = document.getElementById('sectionNextBtn');
    const sec = getCurrentSection();
    if (!hint) return;
    if (!sec) {{
        hint.textContent = `[0 个 SK节点]`;
        if (chip) chip.textContent = 'SK节点 0 / 0';
        if (prevBtn) prevBtn.disabled = true;
        if (nextBtn) nextBtn.disabled = true;
        return;
    }}
    const flag = sec.incomplete ? ' | 未完整采集' : '';
    const span = (sec.start_line > 0 && sec.end_line > 0) ? `L${{sec.start_line}}-L${{sec.end_line}}` : '-';
    const seq = sec.section_seq != null ? `SK#${{sec.section_seq}}` : 'SK#-';
    const modelInstanceCount = sec.model_instance_count || 0;
    const occ = modelInstanceCount > 1 ? ` | 实例:${{modelInstanceCount}}` : '';
    const preferred = findPreferredSectionIndex(getTaskContext());
    const marker = hasTaskContext(getTaskContext()) && preferred === currentSectionIndex ? ' | 当前 SK节点' : '';
    hint.textContent = `${{sec.source_file}}:${{seq}} | ${{span}}${{flag}}${{occ}}${{marker}}`;
    if (chip) chip.textContent = `SK节点 ${{currentSectionIndex + 1}} / ${{sectionsData.length}}`;
    if (prevBtn) prevBtn.disabled = currentSectionIndex <= 0;
    if (nextBtn) nextBtn.disabled = currentSectionIndex >= sectionsData.length - 1;
}}

function syncDetailCollapsedState() {{
    const box = document.getElementById('detailContent');
    const btn = document.getElementById('detailToggleBtn');
    const head = box ? box.closest('.detail-panel')?.querySelector('[data-detail-toggle]') : null;
    const panel = box ? box.closest('.detail-panel') : null;
    if (box) box.classList.toggle('is-collapsed', detailCollapsed);
    if (panel) panel.classList.toggle('is-open', !detailCollapsed);
    if (btn) {{
        btn.textContent = detailCollapsed ? '展开' : '收起';
    }}
    if (head) {{
        head.setAttribute('aria-expanded', detailCollapsed ? 'false' : 'true');
    }}
}}

function toggleDetailCollapsed() {{
    detailCollapsed = !detailCollapsed;
    syncDetailCollapsedState();
}}

function matchesDetailTask(task, queue, rawTaskIdx) {{
    if (detailQueueFilter !== 'all' && queue !== detailQueueFilter) return false;
    if (detailTypeFilter !== 'all' && task.dispatch_type !== detailTypeFilter) return false;
    const q = (detailSearchQuery || '').trim().toLowerCase();
    if (!q) return true;
    const parts = [
        String(queue || ''),
        String(typeLabel(task.dispatch_type) || ''),
        String(updateSemanticLabel(task.dispatch_type) || ''),
        String(rawTaskIdx || ''),
        String(task.sync_type || task.sync_flag || ''),
        String(task.sync_direction || ''),
        String(task.sync_kind || ''),
        String(task.sync_raw_args || task.args || ''),
        ...(Array.isArray(task.entries) ? task.entries.map(v => String(v || '')) : []),
        String(task.kernel_name || ''),
        String(task.node_id || ''),
        String(task.graph_node_id || ''),
        String(task.event_id || ''),
        String(task.node_type || ''),
    ];
    return parts.join(' ').toLowerCase().includes(q);
}}

function getFilteredDetailTasks(sec) {{
    const tasks = [
        ...(sec.aic_tasks || []).map(task => [task, 'AIC']),
        ...(sec.aiv_tasks || []).map(task => [task, 'AIV']),
    ];
    return tasks.filter(([task, queue]) => matchesDetailTask(task, queue, taskIndexOf(task)));
}}

function syncDetailPagerState(totalRows) {{
    const totalPages = Math.max(1, Math.ceil(totalRows / detailPageSize));
    if (detailPage > totalPages) detailPage = totalPages;
    if (detailPage < 1) detailPage = 1;
    const prevBtn = document.getElementById('detailPrevBtn');
    const nextBtn = document.getElementById('detailNextBtn');
    const status = document.getElementById('detailPageStatus');
    const count = document.getElementById('detailRowCount');
    const size = document.getElementById('detailPageSize');
    const jumpInput = document.getElementById('detailJumpInput');
    if (prevBtn) prevBtn.disabled = detailPage <= 1;
    if (nextBtn) nextBtn.disabled = detailPage >= totalPages;
    if (status) status.textContent = '第 ' + detailPage + ' / ' + totalPages + ' 页';
    if (count) count.textContent = '共 ' + totalRows + ' 条';
    if (size) size.value = String(detailPageSize);
    if (jumpInput) jumpInput.max = String(totalPages);
}}

function renderTaskDetails() {{
    const sec = getCurrentSection();
    const body = document.getElementById('detailTableBody');
    const subtitle = document.getElementById('detailSubtitle');
    const empty = document.getElementById('detailEmpty');
    if (!body || !subtitle || !empty) return;

    if (!sec) {{
        body.innerHTML = '';
        empty.style.display = 'block';
        return;
    }}

    const detailName = String(sec.sk_name || '').trim();
    subtitle.textContent = detailName
        ? `${{detailName}} · 当前 SK节点的任务参数和同步语义`
        : '当前 SK节点的任务参数和同步语义';
    const filteredTasks = getFilteredDetailTasks(sec);
    syncDetailPagerState(filteredTasks.length);
    const start = (detailPage - 1) * detailPageSize;
    const pageTasks = filteredTasks.slice(start, start + detailPageSize);
    const rows = pageTasks.map(([task, queue]) => {{
        const rawTaskIdx = taskIndexOf(task);
        const displayIdx = isEvent(task.dispatch_type) && task.node_id
            ? String(task.node_id)
            : String(rawTaskIdx);
        const syncType = task.dispatch_type === 'SYNC' ? (task.sync_type || task.sync_flag || '-') : '-';
        const rawArgs = task.dispatch_type === 'SYNC'
            ? (task.sync_raw_args || task.args || '-')
            : (task.args || '-');
        const entries = Array.isArray(task.entries) && task.entries.length
            ? task.entries.join(', ')
            : '-';
        const funcName = task.kernel_name || '';
        return `
            <tr>
                <td><span class="mono">${{escapeHtml(queue)}}</span></td>
                <td><span class="mono">${{escapeHtml(typeLabel(task.dispatch_type))}}</span></td>
                <td><span class="mono">${{escapeHtml(displayIdx)}}</span></td>
                <td><span class="mono">${{escapeHtml(syncType)}}</span></td>
                <td><span class="mono">${{escapeHtml(rawArgs)}}</span></td>
                <td class="detail-snippet"><span class="mono">${{escapeHtml(entries)}}</span></td>
                <td class="detail-snippet">${{escapeHtml(funcName)}}</td>
            </tr>
        `;
    }}).join('');
    body.innerHTML = rows;
    empty.style.display = rows ? 'none' : 'block';
}}

function getRecommendedTask(sec, ctx) {{
    if (!sec) return null;
    const allTasks = [...(sec.aic_tasks || []), ...(sec.aiv_tasks || [])];
    if (!allTasks.length) return null;
    for (const task of allTasks) {{
        const taskMatch = ctx.taskIndex != null && taskIndexOf(task) === ctx.taskIndex;
        const nodeMatch = ctx.nodeId != null && (task.graph_node_id === ctx.nodeId || task.node_id === ctx.nodeId);
        if (taskMatch || nodeMatch) return task;
    }}
    return allTasks[0];
}}

function updateContextBoard(state) {{
    return;
}}

function buildSectionOptionText(sec, idx) {{
    const aic = sec.aic_count || 0;
    const aiv = sec.aiv_count || 0;
    const total = aic + aiv;
    const suffix = sec.incomplete ? ' [未完整采集]' : '';
    const seq = sec.section_seq != null ? `SK#${{sec.section_seq}}` : 'SK#-';
    const modelInstanceCount = sec.model_instance_count || 0;
    const dup = modelInstanceCount > 1 ? ` | 实例:${{modelInstanceCount}}` : '';
    const subtitle = sec.sk_subtitle ? ` | ${{sec.sk_subtitle}}` : '';
    const rawName = String(sec.sk_name || sec.sk_label || '').trim();
    return `#${{idx + 1}} ${{rawName || 'SK节点'}} | AIC:${{aic}} AIV:${{aiv}} T:${{total}}${{subtitle}} | ${{sec.source_file}}:${{seq}}${{dup}}${{suffix}}`;
}}

function matchesSectionSearch(sec, idx) {{
    const q = (sectionSearchQuery || '').trim().toLowerCase();
    if (!q) return true;
    const bag = [
        `#${{idx + 1}}`,
        String(sec.sk_label || ''),
        String(sec.sk_subtitle || ''),
        String(sec.sk_search_terms || ''),
        String(sec.sk_id || ''),
        String(sec.source_file || ''),
        String(sec.section_seq || ''),
        String(sec.start_line || ''),
        String(sec.end_line || ''),
        String(sec.aic_count || 0),
        String(sec.aiv_count || 0),
    ].join(' ').toLowerCase();
    return bag.includes(q);
}}

function initSectionSelector() {{
    const select = document.getElementById('sectionSelect');
    const prevBtn = document.getElementById('sectionPrevBtn');
    const nextBtn = document.getElementById('sectionNextBtn');
    if (!select) return;
    select.innerHTML = '';
    if (!sectionsData.length) {{
        const opt = document.createElement('option');
        opt.value = '0';
        opt.textContent = '当前没有可用 SK节点';
        select.appendChild(opt);
        select.disabled = true;
        if (prevBtn) prevBtn.disabled = true;
        if (nextBtn) nextBtn.disabled = true;
        return;
    }}

    sectionOptionIndexMap = [];
    for (let i = 0; i < sectionsData.length; i++) {{
        if (!matchesSectionSearch(sectionsData[i], i)) continue;
        const opt = document.createElement('option');
        opt.value = String(sectionOptionIndexMap.length);
        opt.textContent = buildSectionOptionText(sectionsData[i], i);
        select.appendChild(opt);
        sectionOptionIndexMap.push(i);
    }}
    if (!sectionOptionIndexMap.length) {{
        const opt = document.createElement('option');
        opt.value = '0';
        opt.textContent = '没有匹配的 SK节点';
        select.appendChild(opt);
        select.disabled = true;
        return;
    }}
    select.disabled = false;
    const visibleIndex = sectionOptionIndexMap.indexOf(currentSectionIndex);
    select.value = String(visibleIndex >= 0 ? visibleIndex : 0);
    select.addEventListener('change', () => {{
        const mappedIndex = sectionOptionIndexMap[Number(select.value)];
        currentSectionIndex = mappedIndex != null ? mappedIndex : currentSectionIndex;
        detailPage = 1;
        refreshCurrentSectionData();
        refreshHeaderStats();
        refreshSectionHint();
        renderTaskDetails();
        render();
        requestAnimationFrame(() => {{ resetView(); }});
    }});
    if (prevBtn) {{
        prevBtn.onclick = () => switchSection(-1);
    }}
    if (nextBtn) {{
        nextBtn.onclick = () => switchSection(1);
    }}
}}

function switchSection(delta) {{
    if (!sectionsData.length) return;
    const next = currentSectionIndex + delta;
    if (next < 0 || next >= sectionsData.length) return;
    currentSectionIndex = next;
    detailPage = 1;
    const select = document.getElementById('sectionSelect');
    if (select) {{
        const visibleIndex = sectionOptionIndexMap.indexOf(currentSectionIndex);
        if (visibleIndex >= 0) select.value = String(visibleIndex);
    }}
    refreshCurrentSectionData();
    refreshHeaderStats();
    refreshSectionHint();
    renderTaskDetails();
    render();
    requestAnimationFrame(() => {{ resetView(); }});
}}

// ==================== Color helpers ====================
const COLORS_MAP = {{
    'FUNC':         {{ bg: '#16a34a', border: '#166534', text: '#ffffff', tagBg: '#dcfce7', tagText: '#166534' }},
    'PRELOAD':      {{ bg: '#2563eb', border: '#1d4ed8', text: '#ffffff', tagBg: '#dbeafe', tagText: '#1d4ed8' }},
    'SYNC':         {{ bg: '#d97706', border: '#b45309', text: '#ffffff', tagBg: '#ffedd5', tagText: '#9a3412' }},
    'EVENT_NOTIFY': {{ bg: '#be185d', border: '#9d174d', text: '#ffffff', tagBg: '#fce7f3', tagText: '#9d174d' }},
    'EVENT_WAIT':   {{ bg: '#6d28d9', border: '#5b21b6', text: '#ffffff', tagBg: '#ede9fe', tagText: '#5b21b6' }},
    'EVENT_RESET':  {{ bg: '#475569', border: '#334155', text: '#ffffff', tagBg: '#e2e8f0', tagText: '#334155' }},
}};

function getStreamColor(streamIdx) {{
    const n = Number(streamIdx);
    if (!Number.isFinite(n) || n < 0) return null;
    const hue = ((Math.trunc(n) % 12) * 29 + 18) % 360;
    return {{
        bg: `hsl(${{hue}}, 84%, 48%)`,
        border: `hsl(${{hue}}, 78%, 24%)`,
        text: '#fff',
        glow: `hsla(${{hue}}, 90%, 42%, 0.40)`,
    }};
}}

function getColor(type, queue, streamIdxInGraph) {{
    return COLORS_MAP[type] || {{ bg: '#475569', border: '#1e293b', text: '#fff', tagBg: '#e2e8f0', tagText: '#334155' }};
}}

function typeLabel(type) {{
    if (type === 'FUNC') return 'Func';
    if (type === 'PRELOAD') return 'Preload';
    if (type === 'SYNC') return 'Sync';
    if (type === 'EVENT_NOTIFY') return 'Notify';
    if (type === 'EVENT_WAIT') return 'Wait';
    if (type === 'EVENT_RESET') return 'Reset';
    return type;
}}

function updateSemanticLabel(type) {{
    if (type === 'EVENT_NOTIFY') return 'Value Write';
    if (type === 'EVENT_WAIT') return 'Value Wait';
    if (type === 'EVENT_RESET') return 'Value Write';
    return '-';
}}

function shortEventId(eid) {{
    if (!eid) return '';
    const s = String(eid);
    const clean = s.startsWith('0x') ? s.slice(2) : s;
    if (clean.length <= 6) return clean;
    return clean.slice(-6);
}}

function shortSyncFlag(flag) {{
    if (!flag) return 'SYNC';
    const s = String(flag);
    if (s.startsWith('SET_')) {{
        return 'S ' + s.slice(4).replaceAll('_TO_', '->');
    }}
    if (s.startsWith('WAIT_')) {{
        return 'W ' + s.slice(5).replaceAll('_TO_', '->');
    }}
    return s;
}}

function syncKind(flag) {{
    if (!flag) return 'SYNC';
    const s = String(flag);
    if (s.startsWith('SET_')) return 'SET';
    if (s.startsWith('WAIT_')) return 'WAIT';
    return 'SYNC';
}}

function shortTaskMeta(task) {{
    return '';
}}

function measureVisualText(text, kind = 'body') {{
    const s = String(text || '');
    if (!s) return 0;
    let width = 0;
    for (const ch of s) {{
        if (/[\u3400-\u9fff]/.test(ch)) {{
            width += kind === 'mono' ? 11.2 : 10.6;
        }} else if (/[A-Z]/.test(ch)) {{
            width += kind === 'mono' ? 7.2 : 7.0;
        }} else if (/[a-z]/.test(ch)) {{
            width += kind === 'mono' ? 6.5 : 6.1;
        }} else if (/[0-9]/.test(ch)) {{
            width += kind === 'mono' ? 6.7 : 6.3;
        }} else if (/[#:_./-]/.test(ch)) {{
            width += kind === 'mono' ? 6.0 : 5.7;
        }} else {{
            width += kind === 'mono' ? 6.4 : 6.0;
        }}
    }}
    return width;
}}

function fitVisualText(text, maxW, kind = 'body') {{
    const s = String(text || '');
    if (!s) return '';
    if (measureVisualText(s, kind) <= maxW) return s;
    let out = '';
    for (const ch of s) {{
        const next = out + ch;
        if (measureVisualText(next + '..', kind) > maxW) break;
        out = next;
    }}
    return out ? out + '..' : '..';
}}

function getTaskPrimaryText(task) {{
    if (task.dispatch_type === 'FUNC' || task.dispatch_type === 'PRELOAD') {{
        if (task.kernel_name) {{
            return task.kernel_name.length > 26 ? task.kernel_name.slice(0, 24) + '..' : task.kernel_name;
        }}
        return task.dispatch_type === 'FUNC' ? 'Kernel Func' : 'Preload';
    }}
    if (task.dispatch_type === 'SYNC') {{
        const dirText = formatSyncDir(task.sync_direction || '');
        if (dirText) return dirText;
        const kind = syncKind(task.sync_flag || task.sync_type || '');
        if (kind === 'SET') return 'SET';
        if (kind === 'WAIT') return 'WAIT';
        if (kind === 'ALL') return 'ALL';
        return 'SYNC';
    }}
    if (task.dispatch_type === 'EVENT_NOTIFY') return 'Notify';
    if (task.dispatch_type === 'EVENT_WAIT') return 'Wait';
    if (task.dispatch_type === 'EVENT_RESET') return 'Reset';
    return typeLabel(task.dispatch_type);
}}

function getTaskIndexLabel(task) {{
    const idx = taskIndexOf(task);
    if (idx === undefined || idx === null || idx === '') return '';
    return `#${{idx}}`;
}}

function getTaskBadgeText(task) {{
    if (isEvent(task.dispatch_type)) return 'Event';
    return typeLabel(task.dispatch_type);
}}

function isCrossLaneSync(task) {{
    if (task.dispatch_type !== 'SYNC') return false;
    const dir = String(task.sync_direction || '').toUpperCase();
    return dir === 'AIC_TO_AIV' || dir === 'AIV_TO_AIC';
}}

function buildNodeOutline(g, task, x, y, nodeW, nodeH, color) {{
    if (task.dispatch_type === 'PRELOAD') {{
        const cut = 12;
        const d = [
            `M ${{x + cut}} ${{y}}`,
            `L ${{x + nodeW - cut}} ${{y}}`,
            `L ${{x + nodeW}} ${{y + nodeH / 2}}`,
            `L ${{x + nodeW - cut}} ${{y + nodeH}}`,
            `L ${{x + cut}} ${{y + nodeH}}`,
            `L ${{x}} ${{y + nodeH / 2}}`,
            'Z',
        ].join(' ');
        return svgEl('path', {{
            d,
            fill: color.bg,
            stroke: color.border,
            'stroke-width': String(VIS.node.borderWidth),
            'vector-effect': 'non-scaling-stroke',
        }}, g);
    }}
    if (task.dispatch_type === 'SYNC' && isCrossLaneSync(task)) {{
        const cut = 16;
        const d = [
            `M ${{x + cut}} ${{y}}`,
            `L ${{x + nodeW - cut}} ${{y}}`,
            `L ${{x + nodeW}} ${{y + nodeH / 2}}`,
            `L ${{x + nodeW - cut}} ${{y + nodeH}}`,
            `L ${{x + cut}} ${{y + nodeH}}`,
            `L ${{x}} ${{y + nodeH / 2}}`,
            'Z',
        ].join(' ');
        return svgEl('path', {{
            d,
            fill: color.bg,
            stroke: color.border,
            'stroke-width': String(VIS.node.borderWidth),
            'vector-effect': 'non-scaling-stroke',
        }}, g);
    }}
    if (isEvent(task.dispatch_type)) {{
        return svgEl('rect', {{
            x, y, width: nodeW, height: nodeH,
            rx: nodeH / 2, fill: color.bg,
            stroke: color.border, 'stroke-width': String(VIS.node.borderWidth),
            'vector-effect': 'non-scaling-stroke',
        }}, g);
    }}
    return svgEl('rect', {{
        x, y, width: nodeW, height: nodeH,
        rx: VIS.node.radiusLg, fill: color.bg,
        stroke: color.border, 'stroke-width': String(VIS.node.borderWidth),
        'vector-effect': 'non-scaling-stroke',
    }}, g);
}}

function formatSyncDir(dir) {{
    if (!dir) return '';
    return String(dir).replaceAll('_TO_', '->');
}}

function getSyncEdgeTheme(dirRaw) {{
    const dir = String(dirRaw || '').toUpperCase();
    // Prefer low-saturation pastel tones for readability.
    if (dir === 'AIC_TO_AIV') {{
        return {{
            stroke: '#89AFC6',
            glow: 'rgba(137,175,198,0.26)',
            text: '#4E6D83',
            pillStroke: 'rgba(137,175,198,0.45)',
        }};
    }}
    if (dir === 'AIV_TO_AIC') {{
        return {{
            stroke: '#9BBDAA',
            glow: 'rgba(155,189,170,0.26)',
            text: '#557866',
            pillStroke: 'rgba(155,189,170,0.45)',
        }};
    }}
    return {{
        stroke: '#B3AEC7',
        glow: 'rgba(179,174,199,0.24)',
        text: '#6C6687',
        pillStroke: 'rgba(179,174,199,0.40)',
    }};
}}

function isEvent(type) {{
    return type.startsWith('EVENT_');
}}

function taskIndexOf(task) {{
    if (task.task_idx !== undefined && task.task_idx !== null) return task.task_idx;
    return task.stask_idx;
}}

function shouldShow(task) {{
    if (activeFilters.has('EVENT') && isEvent(task.dispatch_type)) return true;
    return activeFilters.has(task.dispatch_type);
}}

function getTaskWidthRange(task) {{
    if (task.dispatch_type === 'FUNC') return {{ min: CFG.kernelMinW, max: CFG.kernelMaxW }};
    if (task.dispatch_type === 'PRELOAD') return {{ min: CFG.preloadMinW, max: CFG.preloadMaxW }};
    return {{ min: CFG.compactMinW, max: CFG.compactMaxW }};
}}

function getTaskWidth(task) {{
    const badgeLabel = getTaskBadgeText(task);
    const primaryText = getTaskPrimaryText(task);
    const metaText = shortTaskMeta(task);
    const taskIndexLabel = getTaskIndexLabel(task);
    const range = getTaskWidthRange(task);
    const badgeW = Math.max(42, Math.ceil(measureVisualText(badgeLabel, 'body') + 16));
    const idxW = taskIndexLabel ? Math.ceil(measureVisualText(taskIndexLabel, 'mono') + 16) : 0;
    const topRowW = 20 + badgeW + (idxW ? idxW + 12 : 0);
    const bodyW = Math.max(
        Math.ceil(measureVisualText(primaryText, 'mono') + 28),
        metaText ? Math.ceil(measureVisualText(metaText, 'body') + 24) : 0,
    );
    const desired = Math.max(topRowW, bodyW);
    return Math.max(range.min, Math.min(range.max, desired));
}}

function getSlotW() {{
    return currentSlotW;
}}

function updateSlotWidth(tasks) {{
    const widest = (tasks || []).reduce((maxW, task) => Math.max(maxW, getTaskWidth(task)), CFG.kernelMinW);
    currentSlotW = Math.max(CFG.compactMinW + CFG.gapX, widest + CFG.gapX);
}}

function getTaskX(task, slotPos, nodeX0) {{
    const slotX = nodeX0 + slotPos * getSlotW();
    const taskW = getTaskWidth(task);
    return slotX + (currentSlotW - CFG.gapX - taskW) / 2;
}}

// ==================== SVG Helpers ====================
function svgEl(tag, attrs, parent) {{
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs || {{}})) {{
        el.setAttribute(k, v);
    }}
    if (parent) parent.appendChild(el);
    return el;
}}

function escapeHtml(s) {{
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}}

function compareSyncTasks(a, b) {{
    const aNode = a.node_id != null ? Number(a.node_id) : -1;
    const bNode = b.node_id != null ? Number(b.node_id) : -1;
    return (aNode - bNode) || (taskIndexOf(a) - taskIndexOf(b)) || ((a.task_id || 0) - (b.task_id || 0));
}}

function collectSyncLinks(aicTasks, aivTasks) {{
    const allSyncs = [];
    for (let i = 0; i < aicTasks.length; i++) {{
        const t = aicTasks[i];
        if (t.dispatch_type === 'SYNC') allSyncs.push({{ ...t, _queuePos: i, _queue: 'aic' }});
    }}
    for (let i = 0; i < aivTasks.length; i++) {{
        const t = aivTasks[i];
        if (t.dispatch_type === 'SYNC') allSyncs.push({{ ...t, _queuePos: i, _queue: 'aiv' }});
    }}

    const buckets = {{}};
    for (const s of allSyncs) {{
        const dir = String(s.sync_direction || '');
        if (!buckets[dir]) buckets[dir] = {{ sets: [], waits: [], alls: {{ aic: [], aiv: [] }} }};
        if (s.sync_kind === 'SET') {{
            buckets[dir].sets.push(s);
        }} else if (s.sync_kind === 'WAIT') {{
            buckets[dir].waits.push(s);
        }} else if (s.sync_kind === 'ALL') {{
            buckets[dir].alls[s._queue].push(s);
        }}
    }}

    const links = [];
    for (const [dir, bucket] of Object.entries(buckets)) {{
        bucket.sets.sort(compareSyncTasks);
        bucket.waits.sort(compareSyncTasks);
        bucket.alls.aic.sort(compareSyncTasks);
        bucket.alls.aiv.sort(compareSyncTasks);

        const pairCount = Math.min(bucket.sets.length, bucket.waits.length);
        for (let i = 0; i < pairCount; i++) {{
            links.push({{
                kind: 'set_wait',
                direction: dir,
                first: bucket.sets[i],
                second: bucket.waits[i],
            }});
        }}

        const allCount = Math.min(bucket.alls.aic.length, bucket.alls.aiv.length);
        for (let i = 0; i < allCount; i++) {{
            links.push({{
                kind: 'all_sync',
                direction: dir || 'ALL',
                first: bucket.alls.aic[i],
                second: bucket.alls.aiv[i],
            }});
        }}
    }}
    return links;
}}

function computeAlignedPositions(aicTasks, aivTasks, links) {{
    const aicPos = Array.from({{ length: aicTasks.length }}, (_, i) => i);
    const aivPos = Array.from({{ length: aivTasks.length }}, (_, i) => i);

    const pairLinks = [];
    for (const link of links) {{
        const firstEntry = link.first;
        const secondEntry = link.second;
        let ai = -1;
        let vi = -1;
        if (firstEntry._queue === 'aic') ai = firstEntry._queuePos;
        if (secondEntry._queue === 'aic') ai = secondEntry._queuePos;
        if (firstEntry._queue === 'aiv') vi = firstEntry._queuePos;
        if (secondEntry._queue === 'aiv') vi = secondEntry._queuePos;
        if (ai >= 0 && vi >= 0) pairLinks.push([ai, vi]);
    }}

    const crossingLinkKeys = new Set();
    for (let i = 0; i < pairLinks.length; i++) {{
        const [ai1, vi1] = pairLinks[i];
        for (let j = i + 1; j < pairLinks.length; j++) {{
            const [ai2, vi2] = pairLinks[j];
            if ((ai1 < ai2 && vi1 > vi2) || (ai1 > ai2 && vi1 < vi2)) {{
                crossingLinkKeys.add(`${{ai1}}:${{vi1}}`);
                crossingLinkKeys.add(`${{ai2}}:${{vi2}}`);
            }}
        }}
    }}

    const alignLinks = pairLinks.filter(([ai, vi]) => !crossingLinkKeys.has(`${{ai}}:${{vi}}`));
    const hasCrossing = crossingLinkKeys.size > 0;

    let changed = true;
    let guard = 0;
    const maxIter = Math.max(8, (aicTasks.length + aivTasks.length) * 4);
    while (changed && guard < maxIter) {{
        guard++;
        changed = false;

        for (const [ai, vi] of alignLinks) {{
            const target = Math.max(aicPos[ai], aivPos[vi]);
            if (aicPos[ai] !== target) {{ aicPos[ai] = target; changed = true; }}
            if (aivPos[vi] !== target) {{ aivPos[vi] = target; changed = true; }}
        }}

        for (let i = 1; i < aicPos.length; i++) {{
            const minV = aicPos[i - 1] + 1;
            if (aicPos[i] < minV) {{ aicPos[i] = minV; changed = true; }}
        }}
        for (let i = 1; i < aivPos.length; i++) {{
            const minV = aivPos[i - 1] + 1;
            if (aivPos[i] < minV) {{ aivPos[i] = minV; changed = true; }}
        }}
    }}

    const maxSlot = Math.max(
        aicPos.length > 0 ? aicPos[aicPos.length - 1] : 0,
        aivPos.length > 0 ? aivPos[aivPos.length - 1] : 0,
    );
    return {{
        aicPos,
        aivPos,
        slotCount: Math.max(1, maxSlot + 1),
        hasCrossing,
        crossingLinkKeys: Array.from(crossingLinkKeys),
    }};
}}

// ==================== Render ====================
function render() {{
    const svg = document.getElementById('mainSvg');
    svg.innerHTML = '';

    // Build defs
    const defs = svgEl('defs', null, svg);

    // Grid pattern
    const gridPat = svgEl('pattern', {{
        id: 'grid', width: '40', height: '40', patternUnits: 'userSpaceOnUse',
    }}, defs);
    svgEl('path', {{ d: 'M 40 0 L 0 0 0 40', fill: 'none', stroke: 'rgba(0,0,0,0.04)', 'stroke-width': '1' }}, gridPat);

    // Build layout
    const aicFiltered = aicData.filter(shouldShow);
    const aivFiltered = aivData.filter(shouldShow);
    updateSlotWidth([...aicFiltered, ...aivFiltered]);
    const syncLinks = collectSyncLinks(aicFiltered, aivFiltered);
    const layout = computeAlignedPositions(aicFiltered, aivFiltered, syncLinks);
    const maxTasks = layout.slotCount;

    const totalW = CFG.padLeft + CFG.streamLabelW + maxTasks * getSlotW() + 96;
    const totalH = CFG.padTop + 2 * CFG.nodeH + CFG.streamGap + 104;

    svg.setAttribute('width', totalW);
    svg.setAttribute('height', totalH);
    svg.setAttribute('viewBox', `0 0 ${{totalW}} ${{totalH}}`);
    svg.setAttribute('shape-rendering', 'crispEdges');
    svg.setAttribute('text-rendering', 'optimizeLegibility');

    // Background
    svgEl('rect', {{ width: totalW, height: totalH, fill: '#f4f7fb' }}, svg);
    svgEl('rect', {{ width: totalW, height: totalH, fill: 'url(#grid)' }}, svg);

    const sec = getCurrentSection();

    // Stream Y positions
    const aicY = CFG.padTop + 6;
    const aivY = CFG.padTop + CFG.nodeH + CFG.streamGap;
    const nodeX0 = CFG.padLeft + CFG.streamLabelW;

    // Stream zone backgrounds
    svgEl('rect', {{
        x: nodeX0 - 28, y: aicY - 18, width: maxTasks * getSlotW() + 56, height: CFG.nodeH + 36,
        rx: VIS.lane.bgRadius, fill: '#eef6fb', stroke: '#c6d9e6', 'stroke-width': '1',
    }}, svg);
    svgEl('rect', {{
        x: nodeX0 - 28, y: aivY - 18, width: maxTasks * getSlotW() + 56, height: CFG.nodeH + 36,
        rx: VIS.lane.bgRadius, fill: '#f5effb', stroke: '#dacaf0', 'stroke-width': '1',
    }}, svg);

    // Stream labels
    const aicLabelBg = svgEl('rect', {{
        x: CFG.padLeft, y: aicY, width: CFG.streamLabelW, height: CFG.nodeH,
        rx: VIS.lane.labelRadius, fill: '#dbeaf5', stroke: '#9dc0d8', 'stroke-width': String(VIS.lane.lineWidth),
    }}, svg);
    const aicLabel = svgEl('text', {{
        x: CFG.padLeft + CFG.streamLabelW / 2, y: aicY + CFG.nodeH / 2 + 5,
        fill: '#315267', 'font-size': String(VIS.node.queueLabelFontSize), 'font-weight': '800',
        'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_UI_STACK}",
        'letter-spacing': '0.5',
    }}, svg);
    aicLabel.textContent = 'AIC 队列';

    const aivLabelBg = svgEl('rect', {{
        x: CFG.padLeft, y: aivY, width: CFG.streamLabelW, height: CFG.nodeH,
        rx: VIS.lane.labelRadius, fill: '#ece1f8', stroke: '#c8b1e1', 'stroke-width': String(VIS.lane.lineWidth),
    }}, svg);
    const aivLabel = svgEl('text', {{
        x: CFG.padLeft + CFG.streamLabelW / 2, y: aivY + CFG.nodeH / 2 + 5,
        fill: '#5c4573', 'font-size': String(VIS.node.queueLabelFontSize), 'font-weight': '800',
        'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_UI_STACK}",
        'letter-spacing': '0.5',
    }}, svg);
    aivLabel.textContent = 'AIV 队列';

    // Queue guide line
    const flowEndX = nodeX0 + maxTasks * getSlotW() + 10;
        svgEl('line', {{
            x1: nodeX0 - 10, y1: aicY + CFG.nodeH / 2,
            x2: flowEndX, y2: aicY + CFG.nodeH / 2,
            stroke: '#bfd2e0', 'stroke-width': String(VIS.edge.widthQueue),
        }}, svg);
        svgEl('line', {{
            x1: nodeX0 - 10, y1: aivY + CFG.nodeH / 2,
            x2: flowEndX, y2: aivY + CFG.nodeH / 2,
            stroke: '#d7c7eb', 'stroke-width': String(VIS.edge.widthQueue),
        }}, svg);

    // Draw cross-queue edges FIRST (behind nodes)
    drawCrossQueueEdges(
        svg,
        syncLinks,
        aicY,
        aivY,
        nodeX0,
        layout.aicPos,
        layout.aivPos,
        new Set(layout.crossingLinkKeys || []),
    );

    // Draw intra-queue flow arrows between consecutive nodes
    drawFlowArrows(svg, aicFiltered, layout.aicPos, aicY, nodeX0);
    drawFlowArrows(svg, aivFiltered, layout.aivPos, aivY, nodeX0);

    // Draw AIC tasks
    for (let i = 0; i < aicFiltered.length; i++) {{
        const task = aicFiltered[i];
        const taskW = getTaskWidth(task);
        const x = getTaskX(task, layout.aicPos[i], nodeX0);
        const y = aicY;
        drawTaskNode(svg, task, x, y, i, taskW);
    }}

    // Draw AIV tasks
    for (let i = 0; i < aivFiltered.length; i++) {{
        const task = aivFiltered[i];
        const taskW = getTaskWidth(task);
        const x = getTaskX(task, layout.aivPos[i], nodeX0);
        const y = aivY;
        drawTaskNode(svg, task, x, y, i, taskW);
    }}

    applyTaskContext();
}}

function drawFlowArrows(svg, tasks, positions, queueY, nodeX0) {{
    for (let i = 0; i < tasks.length - 1; i++) {{
        const w1 = getTaskWidth(tasks[i]);
        const w2 = getTaskWidth(tasks[i + 1]);
        const x1 = getTaskX(tasks[i], positions[i], nodeX0) + w1;
        const x2 = getTaskX(tasks[i + 1], positions[i + 1], nodeX0);
        const cy = queueY + CFG.nodeH / 2;

        svgEl('line', {{
            x1: x1 + 2, y1: cy, x2: x2 - 2, y2: cy,
            stroke: '#cbd5e1', 'stroke-width': String(VIS.edge.widthQueue), 'stroke-linecap': 'round',
            'vector-effect': 'non-scaling-stroke',
        }}, svg);
    }}
}}

function drawTaskNode(svg, task, x, y, idx, nodeW) {{
    const c = getColor(task.dispatch_type, task.queue, task.stream_idx_in_graph);
    const badgeLabel = getTaskBadgeText(task);
    const primaryText = getTaskPrimaryText(task);
    const metaText = shortTaskMeta(task);
    const taskIndexLabel = getTaskIndexLabel(task);
    const g = svgEl('g', {{
        class: 'task-node',
        'data-task-id': idx,
        'data-task-index': taskIndexOf(task),
        'data-node-id': task.graph_node_id,
        'data-x': x,
        'data-y': y,
        'data-queue': task.queue,
        style: 'cursor:pointer;',
    }}, svg);

    const mainRect = buildNodeOutline(g, task, x, y, nodeW, CFG.nodeH, c);

    if (task.dispatch_type === 'SYNC') {{
        const sk = syncKind(task.sync_flag);
        if (sk === 'WAIT') {{
            mainRect.setAttribute('stroke-dasharray', VIS.edge.dashWait);
        }} else if (sk === 'SET') {{
            mainRect.setAttribute('stroke-dasharray', VIS.edge.dashSet);
        }}
    }}

    const rawIdxW = taskIndexLabel ? Math.max(28, Math.ceil(measureVisualText(taskIndexLabel, 'mono') + 14)) : 0;
    const idxW = taskIndexLabel ? Math.min(rawIdxW, Math.max(34, nodeW - 78)) : 0;
    const badgeW = Math.min(
        Math.max(42, Math.ceil(measureVisualText(badgeLabel, 'body') + 16)),
        Math.max(44, nodeW - (idxW ? idxW + 30 : 18)),
    );
    const primaryMaxW = Math.max(42, nodeW - 26);
    const primaryShown = fitVisualText(primaryText, primaryMaxW, 'mono');
    const metaShown = fitVisualText(metaText, Math.max(36, nodeW - 24), 'body');
    svgEl('rect', {{
        x: x + 10, y: y + 8, width: badgeW, height: 18,
        rx: VIS.node.radiusPill, fill: c.tagBg,
    }}, g);
    const badgeText = svgEl('text', {{
        x: x + 10 + badgeW / 2, y: y + 20,
        fill: c.tagText, 'font-size': String(VIS.node.badgeFontSize), 'font-weight': '800',
        'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_UI_STACK}",
        'letter-spacing': '0.3',
    }}, g);
    badgeText.textContent = badgeLabel;

    if (taskIndexLabel) {{
        svgEl('rect', {{
            x: x + nodeW - idxW - 10, y: y + 8, width: idxW, height: 18,
            rx: VIS.node.radiusPill, fill: 'rgba(255,255,255,0.24)', stroke: 'rgba(255,255,255,0.16)', 'stroke-width': '0.8',
        }}, g);
        const idxText = svgEl('text', {{
            x: x + nodeW - idxW / 2 - 10, y: y + 20,
            fill: 'rgba(255,255,255,0.96)', 'font-size': String(VIS.node.badgeMonoFontSize),
            'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_MONO_STACK}",
            'font-weight': '700',
        }}, g);
        idxText.textContent = taskIndexLabel;
    }}

    if (primaryText) {{
        const primary = svgEl('text', {{
            x: x + nodeW / 2, y: task.dispatch_type === 'SYNC' ? (y + 42) : (metaText ? (y + 38) : (y + 42)),
            fill: c.text, 'font-size': String(VIS.node.titleFontSize + 1),
            'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_MONO_STACK}",
            'font-weight': '800',
        }}, g);
        primary.textContent = primaryShown;
    }}

    if (metaText && task.dispatch_type !== 'SYNC') {{
        const meta = svgEl('text', {{
            x: x + nodeW / 2, y: y + CFG.nodeH - 14,
            fill: 'rgba(255,255,255,0.82)', 'font-size': String(VIS.node.subtitleFontSize),
            'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_UI_STACK}",
            'font-weight': '700',
        }}, g);
        meta.textContent = metaShown;
    }}

    // Hover / tooltip events
    g.addEventListener('mouseenter', (e) => showTooltip(e, task));
    g.addEventListener('mousemove', (e) => moveTooltip(e));
    g.addEventListener('mouseleave', hideTooltip);
}}

function applyTaskContext() {{
    const ctx = getTaskContext();
    const sec = getCurrentSection();
    const state = {{
        query: ctx,
        section: sec,
        hit: false,
        relatedCount: 0,
        primaryTaskIndex: null,
    }};
    if (!hasTaskContext(ctx)) return state;

    const groups = Array.from(document.querySelectorAll('.task-node'));
    let primary = null;
    const related = [];

    for (const g of groups) {{
        const taskIndex = parseInt(g.dataset.taskIndex || '-1', 10);
        const nodeId = parseInt(g.dataset.nodeId || '-1', 10);
        const taskMatch = ctx.taskIndex != null && taskIndex === ctx.taskIndex;
        const nodeMatch = ctx.nodeId != null && nodeId === ctx.nodeId;
        if (taskMatch && !primary) primary = g;
        if (taskMatch || nodeMatch) related.push(g);
    }}
    if (!primary && related.length) primary = related[0];
    if (!primary) return state;
    state.hit = true;
    state.relatedCount = related.length;
    state.primaryTaskIndex = parseInt(primary.dataset.taskIndex || '-1', 10);
    return state;
}}

function updateTaskFocusSummary(state) {{
    return;
}}

function drawCrossQueueEdges(svg, links, aicY, aivY, nodeX0, aicPos, aivPos, crossingLinkKeys) {{
    for (let pairIdx = 0; pairIdx < links.length; pairIdx++) {{
        const link = links[pairIdx];
        const firstEntry = link.first;
        const secondEntry = link.second;
        const firstPos = firstEntry._queue === 'aic' ? aicPos[firstEntry._queuePos] : aivPos[firstEntry._queuePos];
        const secondPos = secondEntry._queue === 'aic' ? aicPos[secondEntry._queuePos] : aivPos[secondEntry._queuePos];
        const firstW = getTaskWidth(firstEntry);
        const secondW = getTaskWidth(secondEntry);
        const firstNodeX = getTaskX(firstEntry, firstPos, nodeX0) + firstW / 2;
        const secondNodeX = getTaskX(secondEntry, secondPos, nodeX0) + secondW / 2;
        const firstY = firstEntry._queue === 'aic'
            ? (aicY + CFG.nodeH + CFG.edgeEndpointPad)
            : (aivY - CFG.edgeEndpointPad);
        const secondY = secondEntry._queue === 'aic'
            ? (aicY + CFG.nodeH + CFG.edgeEndpointPad)
            : (aivY - CFG.edgeEndpointPad);
        const ai = firstEntry._queue === 'aic' ? firstEntry._queuePos : secondEntry._queuePos;
        const vi = firstEntry._queue === 'aiv' ? firstEntry._queuePos : secondEntry._queuePos;
        const key = `${{ai}}:${{vi}}`;
        const isCrossingLink = crossingLinkKeys.has(key);
        const dirRaw = String(link.direction || '');
        const dirLabel = link.kind === 'all_sync' ? 'ALL' : formatSyncDir(dirRaw);
        const edgeTheme = getSyncEdgeTheme(dirRaw);
        const midY = (firstY + secondY) / 2 + ((pairIdx % 3) - 1) * 6;
        const railX = (firstNodeX + secondNodeX) / 2;
        const pathD = isCrossingLink
            ? `M ${{firstNodeX}} ${{firstY}} C ${{firstNodeX}} ${{midY}}, ${{secondNodeX}} ${{midY}}, ${{secondNodeX}} ${{secondY}}`
            : `M ${{firstNodeX}} ${{firstY}} L ${{railX}} ${{firstY}} L ${{railX}} ${{secondY}} L ${{secondNodeX}} ${{secondY}}`;

        svgEl('path', {{
            d: pathD,
            fill: 'none', stroke: edgeTheme.stroke,
            'stroke-width': String(link.kind === 'all_sync' ? VIS.edge.widthAllSync : VIS.edge.widthSync),
            'stroke-linecap': 'round',
            'stroke-dasharray': link.kind === 'all_sync' ? '' : VIS.edge.dashSync,
            opacity: '0.92',
            style: link.kind === 'all_sync' ? '' : 'animation: dashFlow 1s linear infinite;',
            'vector-effect': 'non-scaling-stroke',
        }}, svg);

        const labelX = isCrossingLink ? (firstNodeX + secondNodeX) / 2 : railX;
        const labelY = isCrossingLink ? midY - 2 : (firstY + secondY) / 2 - 2;
        const labelWidth = Math.max(34, dirLabel.length * 7 + 14);
        svgEl('rect', {{
            x: labelX - labelWidth / 2, y: labelY - 11, width: labelWidth, height: 18,
            rx: VIS.edge.labelRadiusMd, fill: 'rgba(255,255,255,0.96)', stroke: edgeTheme.pillStroke, 'stroke-width': '1',
            'vector-effect': 'non-scaling-stroke',
        }}, svg);
        const labelText = svgEl('text', {{
            x: labelX, y: labelY,
            fill: edgeTheme.text, 'font-size': String(VIS.edge.labelFontSize), 'font-weight': '600',
            'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_UI_STACK}",
            opacity: '0.96',
        }}, svg);
        labelText.textContent = dirLabel;

        if (link.kind === 'all_sync') {{
            svgEl('circle', {{
                cx: firstNodeX, cy: firstY, r: '4',
                fill: edgeTheme.stroke, opacity: '0.66',
            }}, svg);
            svgEl('circle', {{
                cx: secondNodeX, cy: secondY, r: '4',
                fill: edgeTheme.stroke, opacity: '0.66',
            }}, svg);
            continue;
        }}

        svgEl('circle', {{
            cx: firstNodeX, cy: firstY, r: '4',
            fill: edgeTheme.stroke, opacity: '0.82',
        }}, svg);
        svgEl('circle', {{
            cx: secondNodeX, cy: secondY, r: '4',
            fill: '#ffffff', stroke: edgeTheme.stroke, 'stroke-width': String(VIS.node.borderWidth), opacity: '0.92',
        }}, svg);

        const setNodeX = firstNodeX;
        const waitNodeX = secondNodeX;
        const setLabelY = firstEntry._queue === 'aic' ? (aicY + CFG.nodeH + 18) : (aivY - 10);
        const waitLabelY = secondEntry._queue === 'aic' ? (aicY + CFG.nodeH + 18) : (aivY - 10);

        svgEl('rect', {{
            x: setNodeX - 15, y: setLabelY - 9, width: 30, height: 14,
            rx: VIS.edge.labelRadiusSm, fill: 'rgba(255,255,255,0.96)', stroke: edgeTheme.pillStroke, 'stroke-width': '1',
        }}, svg);
        const setText = svgEl('text', {{
            x: setNodeX, y: setLabelY + 1,
            fill: edgeTheme.text, 'font-size': String(VIS.edge.labelFontSizeSm), 'font-weight': '700',
            'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_UI_STACK}",
        }}, svg);
        setText.textContent = 'SET';

        svgEl('rect', {{
            x: waitNodeX - 17, y: waitLabelY - 9, width: 34, height: 14,
            rx: VIS.edge.labelRadiusSm, fill: 'rgba(255,255,255,0.96)', stroke: edgeTheme.pillStroke, 'stroke-width': '1',
        }}, svg);
        const waitText = svgEl('text', {{
            x: waitNodeX, y: waitLabelY + 1,
            fill: edgeTheme.text, 'font-size': String(VIS.edge.labelFontSizeSm), 'font-weight': '700',
            'text-anchor': 'middle', 'font-family': "{VISUAL_FONT_UI_STACK}",
        }}, svg);
        waitText.textContent = 'WAIT';
    }}
}}

// ==================== Tooltip ====================
function showTooltip(e, task) {{
    const tt = document.getElementById('tooltip');
    const c = getColor(task.dispatch_type, task.queue, task.stream_idx_in_graph);
    const label = typeLabel(task.dispatch_type);
    const queueColor = task.queue === 'aic' ? '#5f8198' : '#7f6b92';
    const queueBg = task.queue === 'aic' ? 'rgba(111,151,177,0.14)' : 'rgba(154,132,175,0.14)';

    let html = `<div class="tt-header">
        <div class="tt-title" style="color:${{c.bg}}">${{escapeHtml(label)}} Seq#${{task.task_id}}</div>
        <span class="tt-queue" style="background:${{queueBg}};color:${{queueColor}}">${{task.queue.toUpperCase()}}</span>
    </div>`;

    html += `<div class="tt-row"><span class="tt-label">TaskIdx:</span><span class="tt-value">${{taskIndexOf(task)}}</span></div>`;
    html += `<div class="tt-row"><span class="tt-label">Type:</span><span class="tt-value">${{task.dispatch_type}}</span></div>`;
    if (task.stream_id !== undefined && task.stream_id !== null && task.stream_id !== -1) {{
        html += `<div class="tt-row"><span class="tt-label">StreamId:</span><span class="tt-value">${{task.stream_id}}</span></div>`;
    }} else if (task.stream_idx_in_graph !== undefined && task.stream_idx_in_graph !== null && task.stream_idx_in_graph !== -1) {{
        html += `<div class="tt-row"><span class="tt-label">GraphStreamIdx:</span><span class="tt-value">${{task.stream_idx_in_graph}}</span></div>`;
    }}

    if (task.dispatch_type === 'FUNC' || task.dispatch_type === 'PRELOAD') {{
        const kernelDisplay = (task.kernel_info && task.kernel_info.funcName) ? task.kernel_info.funcName : task.kernel_name;
        if (kernelDisplay) html += `<div class="tt-row"><span class="tt-label">FuncName:</span><span class="tt-value">${{escapeHtml(kernelDisplay)}}</span></div>`;
        if (task.num_blocks) html += `<div class="tt-row"><span class="tt-label">Blocks:</span><span class="tt-value">${{task.num_blocks}}</span></div>`;
        if (task.kernel_info && task.kernel_info.funcName) {{
            html += `<div class="tt-row"><span class="tt-label">KernelType:</span><span class="tt-value">${{escapeHtml(task.kernel_info.kernelType)}}</span></div>`;
            if (task.kernel_info.cubeNum || task.kernel_info.vecNum) {{
                html += `<div class="tt-row"><span class="tt-label">Shape:</span><span class="tt-value">C:${{task.kernel_info.cubeNum||0}} / V:${{task.kernel_info.vecNum||0}}</span></div>`;
            }}
        }}
    }} else if (task.dispatch_type === 'SYNC') {{
        html += `<div class="tt-row"><span class="tt-label">SyncType:</span><span class="tt-value">${{task.sync_type || task.sync_flag}}</span></div>`;
        html += `<div class="tt-row"><span class="tt-label">Direction:</span><span class="tt-value">${{task.sync_direction || '-'}}</span></div>`;
        html += `<div class="tt-row"><span class="tt-label">Kind:</span><span class="tt-value">${{task.sync_kind || syncKind(task.sync_flag)}}</span></div>`;
        html += `<div class="tt-row"><span class="tt-label">Args:</span><span class="tt-value">${{task.sync_raw_args || '-'}}</span></div>`;
    }} else if (isEvent(task.dispatch_type)) {{
        html += `<div class="tt-row"><span class="tt-label">EventID:</span><span class="tt-value">${{task.event_id}}</span></div>`;
        html += `<div class="tt-row"><span class="tt-label">ShortID:</span><span class="tt-value">${{shortEventId(task.event_id)}}</span></div>`;
        html += `<div class="tt-row"><span class="tt-label">NodeID:</span><span class="tt-value">${{task.node_id}}</span></div>`;
        html += `<div class="tt-row"><span class="tt-label">Args:</span><span class="tt-value">${{task.args || '-'}}</span></div>`;
        if (task.node_type) {{
            html += `<div class="tt-row"><span class="tt-label">NodeType:</span><span class="tt-value">${{task.node_type}}</span></div>`;
        }}
    }}

    tt.innerHTML = html;
    tt.style.display = 'block';
    tt.style.borderColor = c.glow.replace('0.25', '0.35');
    moveTooltip(e);
}}

function moveTooltip(e) {{
    const tt = document.getElementById('tooltip');
    let x = e.clientX + 16;
    let y = e.clientY + 16;
    const rect = tt.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 10) x = e.clientX - rect.width - 10;
    if (y + rect.height > window.innerHeight - 10) y = e.clientY - rect.height - 10;
    if (x < 10) x = 10;
    if (y < 10) y = 10;
    tt.style.left = x + 'px';
    tt.style.top = y + 'px';
}}

function hideTooltip() {{
    document.getElementById('tooltip').style.display = 'none';
}}

// ==================== Zoom / Pan ====================
window.__SK_VIEWPORT_CONFIG = {{
    containerId: 'svgContainer',
    svgId: 'mainSvg',
    minScale: 0.34,
    maxScale: 2.2,
    topPad: 24,
    sidePad: 28,
    fitMode: 'max',
}};
{COMMON_SVG_VIEWPORT_JS}
{COMMON_KEYBOARD_NAV_JS}

// ==================== Filter ====================
function toggleFilter(btn) {{
    const type = btn.getAttribute('data-type');
    if (activeFilters.has(type)) {{
        activeFilters.delete(type);
        btn.classList.remove('active');
    }} else {{
        activeFilters.add(type);
        btn.classList.add('active');
    }}
    render();
}}

// ==================== Init ====================
const preferredSectionIndex = findPreferredSectionIndex(getTaskContext());
if (preferredSectionIndex >= 0) currentSectionIndex = preferredSectionIndex;
const detailToggleBtn = document.getElementById('detailToggleBtn');
const detailToggleHead = document.querySelector('#detailContent') ? document.querySelector('#detailContent').closest('.detail-panel')?.querySelector('[data-detail-toggle]') : null;
const detailPrevBtn = document.getElementById('detailPrevBtn');
const detailNextBtn = document.getElementById('detailNextBtn');
const detailPageSizeSelect = document.getElementById('detailPageSize');
const detailJumpInput = document.getElementById('detailJumpInput');
const detailJumpBtn = document.getElementById('detailJumpBtn');
const sectionSearchInput = document.getElementById('sectionSearchInput');
const kbdModeSelect = document.getElementById('kbdModeSelect');
if (detailToggleBtn) detailToggleBtn.onclick = (ev) => {{ ev.stopPropagation(); toggleDetailCollapsed(); }};
if (detailToggleHead) {{
    detailToggleHead.onclick = (ev) => {{
        const target = ev.target;
        if (target && target.closest('input, select, button, a, textarea')) return;
        toggleDetailCollapsed();
    }};
    detailToggleHead.onkeydown = (ev) => {{
        if (ev.key !== 'Enter' && ev.key !== ' ') return;
        ev.preventDefault();
        toggleDetailCollapsed();
    }};
}}
const detailSearchInput = document.getElementById('detailSearchInput');
const detailQueueSelect = document.getElementById('detailQueueFilter');
const detailTypeSelect = document.getElementById('detailTypeFilter');
if (detailSearchInput) {{
    detailSearchInput.value = detailSearchQuery;
    detailSearchInput.oninput = (ev) => {{
        detailSearchQuery = ev.target.value || '';
        detailPage = 1;
        renderTaskDetails();
    }};
}}
if (detailQueueSelect) {{
    detailQueueSelect.onchange = (ev) => {{
        detailQueueFilter = ev.target.value || 'all';
        detailPage = 1;
        renderTaskDetails();
    }};
}}
if (detailTypeSelect) {{
    detailTypeSelect.onchange = (ev) => {{
        detailTypeFilter = ev.target.value || 'all';
        detailPage = 1;
        renderTaskDetails();
    }};
}}
if (detailPrevBtn) {{
    detailPrevBtn.onclick = () => {{
        if (detailPage > 1) {{
            detailPage -= 1;
            renderTaskDetails();
        }}
    }};
}}
if (detailNextBtn) {{
    detailNextBtn.onclick = () => {{
        detailPage += 1;
        renderTaskDetails();
    }};
}}
if (detailPageSizeSelect) {{
    detailPageSizeSelect.onchange = (ev) => {{
        detailPageSize = parseInt(ev.target.value || '20', 10) || 20;
        detailPage = 1;
        renderTaskDetails();
    }};
}}
const jumpDetailPage = () => {{
    if (!detailJumpInput) return;
    const target = parseInt(detailJumpInput.value || '', 10);
    if (!Number.isFinite(target)) return;
    detailPage = target;
    renderTaskDetails();
}};
if (detailJumpBtn) {{
    detailJumpBtn.onclick = jumpDetailPage;
}}
if (detailJumpInput) {{
    detailJumpInput.onkeydown = (ev) => {{
        if (ev.key === 'Enter') jumpDetailPage();
    }};
}}
if (sectionSearchInput) {{
    sectionSearchInput.value = sectionSearchQuery;
    sectionSearchInput.oninput = (ev) => {{
        sectionSearchQuery = ev.target.value || '';
        initSectionSelector();
        refreshSectionHint();
    }};
}}
if (kbdModeSelect) {{
    kbdModeSelect.value = keyboardNavMode;
    kbdModeSelect.onchange = (ev) => {{
        keyboardNavMode = ev.target.value || 'wasd_zoom';
        syncKeyboardModeControl();
    }};
}}
syncDetailCollapsedState();
syncKeyboardModeControl();
initSectionSelector();
refreshCurrentSectionData();
refreshHeaderStats();
refreshSectionHint();
renderTaskDetails();
render();
attachViewportInteractions();
attachKeyboardNav({{
    previous: () => switchSection(-1),
    next: () => switchSection(1),
    panStep: 80,
}});
requestAnimationFrame(() => {{ resetView(); }});
</script>
</body>
</html>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    _emit(f"Generated HTML: {output_path} ({os.path.getsize(output_path)} bytes)")


# ======================== Main ========================


def main():
    parser = argparse.ArgumentParser(
        description="SK Task Queue Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 模式1：直接指定库文件\n"
            "  python sk_task_queue_visualizer.py \\\n"
            '    --scope-library "$SCOPE_LIBRARY" \\\n'
            '    --graph-library "$GRAPH_LIBRARY" -o task-queue-graph.html\n'
            "  # 模式2：给定目录，自动搜索匹配库对\n"
            '  python sk_task_queue_visualizer.py "$RUN_OR_RESULT_DIR" -o task-queue-graph.html\n'
        ),
    )
    parser.add_argument(
        "log_dir_or_file",
        nargs="?",
        default=".",
        help="目录（自动查找 scope-library.json 和 graph-library.json），或直接指定目录路径",
    )
    parser.add_argument(
        "--scope-library",
        help="直接指定 scope-library.json 路径。和 --graph-library 必须同时提供。",
    )
    parser.add_argument(
        "--graph-library",
        help="直接指定 graph-library.json 路径。和 --scope-library 必须同时提供。",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=None,
        help="输出 HTML 文件路径（默认 task-queue-graph.html）",
    )
    parser.add_argument(
        "output", nargs="?", default=None, help="兼容旧用法，推荐使用 --output"
    )
    args = parser.parse_args()

    log_path = args.log_dir_or_file
    output_path = args.output_path or args.output or "task-queue-graph.html"

    if (args.scope_library is None) ^ (args.graph_library is None):
        _emit("Error: --scope-library 和 --graph-library 必须同时提供。")
        sys.exit(1)

    if args.scope_library and args.graph_library:
        scope_library = args.scope_library
        graph_library = args.graph_library
    else:
        try:
            scope_library, graph_library, _ = _find_library_pair(log_path)
        except FileNotFoundError as exc:
            _emit(exc)
            sys.exit(1)

    if not os.path.isfile(scope_library):
        _emit(f"Error: scope 库不存在: {scope_library}")
        sys.exit(1)
    if not os.path.isfile(graph_library):
        _emit(f"Error: graph 库不存在: {graph_library}")
        sys.exit(1)

    source = TaskQueueLibrarySource(scope_library, graph_library)
    result = TaskQueueModel.from_libraries(source).to_dict()
    if result is None:
        sys.exit(1)

    stats = result.get("stats", {})
    _emit(
        "Summary: sections=%d, source_files=%d"
        % (
            int(stats.get("sections_deduped", 0)),
            int(stats.get("files_scanned", 1)),
        )
    )

    generate_html(result, output_path)


if __name__ == "__main__":
    main()
