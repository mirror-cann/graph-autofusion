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
SK Node Tracing Parser - SuperKernel 模型整图节点可视化工具

解析 SuperKernel 日志中 UpdateNodeScopeBitFlags 阶段的节点信息，
生成 Chrome chrome://tracing/ 可解析的 JSON 文件。

用法:
    python sk_node_tracing.py <log目录或文件> [-o OUTPUT] [--help]

输出:
    默认输出到 <input>_tracing.json，可用 Chrome 浏览器打开 chrome://tracing/ 加载查看。

节点类型:
    Kernel       - 计算内核节点
    EventNotify  - 事件通知节点（发出信号）
    EventWait    - 事件等待节点（接收信号）
    EventReset   - 事件复位节点
    MemoryWrite  - 内存写入节点
    MemoryWait   - 内存等待节点

连边规则:
    - 同一 streamId 的节点按 nodeIdxInStream 顺序排列在同一泳道
    - EventNotify -> EventWait: 相同 eventId 跨流连线
    - EventReset: 与相同 eventId 的 Notify/Wait 关联
    - MemoryWrite -> MemoryWait: 相同 eventId 跨流连线
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

from sk_library_extractor import (
    collect_update_model_instance_reports,
    find_model_dir,
)


# ============================================================
# 节点类型颜色映射
# ============================================================
# 使用 hex 颜色值 (通过 color 字段而非 cname，避免 cname 受限)
NODE_TYPE_COLORS = {
    "Kernel": "#22c55e",  # 绿色
    "EventNotify": "#f97316",  # 橙色
    "EventWait": "#3b82f6",  # 蓝色
    "EventReset": "#ef4444",  # 红色
    "MemoryWrite": "#a855f7",  # 紫色
    "MemoryWait": "#14b8a6",  # 青色
}

NODE_TYPE_CATEGORIES = {
    "Kernel": "kernel",
    "EventNotify": "event_notify",
    "EventWait": "event_wait",
    "EventReset": "event_reset",
    "MemoryWrite": "memory_write",
    "MemoryWait": "memory_wait",
}

TRACE_PROCESS_ID = 1


def _emit(
    message: object = "",
    *,
    file=None,
    end: str = "\n",
    flush: bool = False,
) -> None:
    stream = sys.stdout if file is None else file
    stream.write(f"{message}{end}")
    if flush:
        stream.flush()


NODE_LIBRARY_DISPLAY_TYPES = {
    "KERNEL": "Kernel",
    "EVENT_NOTIFY": "EventNotify",
    "EVENT_WAIT": "EventWait",
    "EVENT_RESET": "EventReset",
    "MEMORY_WRITE": "MemoryWrite",
    "MEMORY_WAIT": "MemoryWait",
}


# ============================================================
# 正则表达式 - 支持新旧两种日志格式
# ============================================================

# 新版格式: Processed node [nodeId:0, streamId:11, streamIdxInGraph:2,
# nodeIdxInStream:0, EventNotify(eventId:0xfffd80cd2b00, eventFlag:0x1)]: type=NOTIFY
RE_NODE_NEW = re.compile(
    r"\[nodeId:(\d+),\s*streamId:(\d+),\s*streamIdxInGraph:(\d+),\s*nodeIdxInStream:(\d+),\s*"
    r"(?:EventNotify|EventWait|EventReset|MemoryWrite|MemoryWait|Kernel name)[^]]*\]"
    r".*?:\s*type=(\w+)"
)

# 新版详细提取 - 分块解析各字段
RE_NODEID = re.compile(r"nodeId:(\d+)")
RE_STREAMID = re.compile(r"streamId:(\d+)")
RE_STREAM_IDX = re.compile(r"streamIdxInGraph:(\d+)")
RE_NODE_IDX = re.compile(r"nodeIdxInStream:(\d+)")

# 事件/内存信息: EventNotify(...) / MemoryWrite(...) / MemoryWait(...)
RE_EVENT = re.compile(r"(EventNotify|EventWait|EventReset|MemoryWrite|MemoryWait)\(([^)]*)\)")
RE_EVENT_ID = re.compile(r"eventId:(0x[0-9a-fA-F]+)")
RE_EVENT_FLAG = re.compile(r"eventFlag:(0x[0-9a-fA-F]+)")

# Kernel 信息: Kernel name:static_kernel_xxx
RE_KERNEL = re.compile(r"Kernel name:([^\]:]+)")
RE_KERNEL_INFOS_FUNC = re.compile(r"KernelInfos\{funcName:([^,}]+)")

# type 字段
RE_TYPE = re.compile(r"type=(\w+)")

# 旧版格式: Processed node [nodeId:0, streamIdxInGraph:2, nodeIdxInStream:0] - Event:Wait(eventId:0xfffd60cd0de0): type=1
RE_NODE_OLD = re.compile(
    r"\[nodeId:(\d+),\s*streamIdxInGraph:(\d+),\s*nodeIdxInStream:(\d+)\]\s*-\s*"
    r"(?:Event|Kernel):[^\)]+?(?:\(eventId:([^\)]+))?\)"
    r".*?:\s*type=(\d+)"
)

# 旧版事件子类型: Event:Wait, Event:Notify 等
RE_OLD_EVENT_SUBTYPE = re.compile(r"Event:(Wait|Notify|Reset)")

# 旧版 type 编号 -> 类型名映射
OLD_TYPE_MAP = {
    "0": "Kernel",
    "1": "EventNotify",
    "2": "EventWait",
    "3": "EventReset",
    "4": "MemoryWrite",
    "5": "MemoryWait",
}

# 起止标记
RE_START = re.compile(r"(?:\[SK\])?\[UpdateNodeScopeBitFlags\]\s+Starting UpdateNodeScopeBitFlags")
RE_END = re.compile(r"(?:\[SK\])?\[UpdateNodeScopeBitFlags\]\s+UpdateNodeScopeBitFlags completed")
RE_PROCESSED = re.compile(r"(?:\[SK\])?\[UpdateNodeScopeBitFlags\]\s+Processed node")


# ============================================================
# 解析函数
# ============================================================


def parse_node_line_new(line):
    """解析新版日志格式的节点行"""
    # 提取基本字段
    m_nodeid = RE_NODEID.search(line)
    m_streamid = RE_STREAMID.search(line)
    m_streamidx = RE_STREAM_IDX.search(line)
    m_nodeidx = RE_NODE_IDX.search(line)
    m_type = RE_TYPE.search(line)

    if not all([m_nodeid, m_streamid, m_streamidx, m_nodeidx, m_type]):
        return None

    node = {
        "nodeId": int(m_nodeid.group(1)),
        "streamId": int(m_streamid.group(1)),
        "streamIdxInGraph": int(m_streamidx.group(1)),
        "nodeIdxInStream": int(m_nodeidx.group(1)),
        "type": m_type.group(1),
    }

    # 确定显示类型名
    type_name = node["type"]
    if type_name == "NOTIFY":
        node["displayType"] = "EventNotify"
    elif type_name == "WAIT":
        node["displayType"] = "EventWait"
    elif type_name == "RESET":
        node["displayType"] = "EventReset"
    elif type_name == "KERNEL":
        node["displayType"] = "Kernel"
    elif type_name == "MEMORY_WRITE":
        node["displayType"] = "MemoryWrite"
    elif type_name == "MEMORY_WAIT":
        node["displayType"] = "MemoryWait"
    else:
        node["displayType"] = type_name

    # 提取事件信息
    m_event = RE_EVENT.search(line)
    if m_event:
        node["eventType"] = m_event.group(1)  # EventNotify/EventWait/...
        inner = m_event.group(2)
        m_eid = RE_EVENT_ID.search(inner)
        if m_eid:
            node["eventId"] = m_eid.group(1)
        m_eflag = RE_EVENT_FLAG.search(inner)
        if m_eflag:
            node["eventFlag"] = m_eflag.group(1)
    else:
        # Kernel
        m_kern = RE_KERNEL.search(line)
        if m_kern:
            node["kernelName"] = m_kern.group(1).strip()
        else:
            m_kinfo = RE_KERNEL_INFOS_FUNC.search(line)
            if m_kinfo:
                node["kernelName"] = m_kinfo.group(1).strip()

    return node


def parse_node_line_old(line):
    """解析旧版日志格式的节点行"""
    m = RE_NODE_OLD.search(line)
    if not m:
        return None

    # 提取基本字段
    m_nodeid = RE_NODEID.search(line)
    m_streamidx = RE_STREAM_IDX.search(line)
    m_nodeidx = RE_NODE_IDX.search(line)

    # group(4) = eventId, group(5) = type code
    event_id = m.group(4).strip() if m.group(4) else ""
    type_code = m.group(5)

    node = {
        "nodeId": int(m_nodeid.group(1)) if m_nodeid else -1,
        "streamId": int(m_streamidx.group(1)) if m_streamidx else -1,
        "streamIdxInGraph": int(m_streamidx.group(1)) if m_streamidx else -1,
        "nodeIdxInStream": int(m_nodeidx.group(1)) if m_nodeidx else -1,
        "type": type_code,
    }

    # type 编号转名称
    type_name = OLD_TYPE_MAP.get(type_code, f"Unknown({type_code})")
    node["displayType"] = type_name

    # 子类型判断: 从 "Event:Wait" / "Event:Notify" / "Event:Reset" 中提取
    m_sub = RE_OLD_EVENT_SUBTYPE.search(line)
    if m_sub:
        sub = m_sub.group(1)
        if sub == "Notify":
            node["eventType"] = "EventNotify"
        elif sub == "Wait":
            node["eventType"] = "EventWait"
        elif sub == "Reset":
            node["eventType"] = "EventReset"

    if event_id:
        node["eventId"] = event_id

    # 旧版 Kernel 名字
    if type_name == "Kernel":
        m_kern = re.search(r"Kernel:([^\]:]+)", line)
        if m_kern:
            node["kernelName"] = m_kern.group(1).strip()

    return node


def short_kernel_name(name):
    """提取 kernel 短名称用于显示"""
    if not name:
        return ""
    # static_kernel_XXX_YYY_hash_d0__kernel0 -> XXX_YYY
    m = re.match(r"static_kernel_([A-Za-z0-9_]+)_[0-9a-f]{16,}_\d+_d\d+", name)
    if m:
        return m.group(1)
    return name[:40]


def short_event_id(eid):
    """截取 eventId 后 4 位十六进制用于显示"""
    if not eid:
        return ""
    eid_clean = eid.replace("0x", "")
    return eid_clean[-4:] if len(eid_clean) >= 4 else eid_clean


def parse_log_file(filepath):
    """
    解析单个日志文件，提取 UpdateNodeScopeBitFlags 阶段的节点信息。
    返回 (nodes_list, source_file) 元组。
    """
    nodes = []
    in_section = False
    saw_start = False
    saw_processed = False

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")

                # 检测起始
                if RE_START.search(line):
                    in_section = True
                    saw_start = True
                    continue

                # 检测结束
                if in_section and RE_END.search(line):
                    break

                processed = RE_PROCESSED.search(line)
                if processed:
                    saw_processed = True

                # 标准模式: 有明确 section 时只处理 section 内的数据
                if saw_start and not in_section:
                    continue

                # fallback 模式: 没有明确起止标记，但文件里本身就只有 processed node
                if not processed:
                    continue

                # 尝试新版格式
                node = parse_node_line_new(line)
                if not node:
                    # 尝试旧版格式
                    node = parse_node_line_old(line)

                if node:
                    nodes.append(node)
    except Exception as e:
        _emit(f"  [WARN] 读取文件出错: {e}", file=sys.stderr)

    if not nodes and saw_processed and os.path.basename(filepath) == "sk_node_detail.log":
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not RE_PROCESSED.search(line):
                        continue
                    node = parse_node_line_new(line) or parse_node_line_old(line)
                    if node:
                        nodes.append(node)
        except Exception as e:
            _emit(f"  [WARN] fallback 读取文件出错: {e}", file=sys.stderr)

    return nodes


def collect_log_files(input_path):
    """收集要解析的日志文件列表"""
    if os.path.isfile(input_path):
        return [input_path]

    files = []
    if os.path.isdir(input_path):
        for root, _, names in os.walk(input_path):
            for fname in names:
                if fname.endswith(".log") or fname.endswith(".txt") or fname.startswith("plog-"):
                    files.append(os.path.join(root, fname))
    files = sorted(set(files))
    return files


def _display_type_from_node_library(row):
    node_type = str(row.get("node_type") or "").upper()
    if node_type in NODE_LIBRARY_DISPLAY_TYPES:
        return NODE_LIBRARY_DISPLAY_TYPES[node_type]
    event_type = str(row.get("event_type") or "")
    if event_type in {
        "EventNotify",
        "EventWait",
        "EventReset",
        "MemoryWrite",
        "MemoryWait",
    }:
        return event_type
    if row.get("func_name"):
        return "Kernel"
    return node_type or "Unknown"


def _node_from_library_row(row):
    display_type = _display_type_from_node_library(row)
    event_type = row.get("event_type") if isinstance(row.get("event_type"), str) else ""
    kernel_name = row.get("func_name") if isinstance(row.get("func_name"), str) else ""
    return {
        "nodeId": int(row.get("node_id", -1)),
        "streamId": int(row.get("stream_id", -1)),
        "streamIdxInGraph": int(row.get("stream_idx_in_graph", -1)),
        "nodeIdxInStream": int(row.get("node_idx_in_stream", -1)),
        "type": str(row.get("node_type") or display_type),
        "displayType": display_type,
        "eventType": event_type,
        "eventId": row.get("event_id"),
        "eventFlag": row.get("event_flag"),
        "kernelName": kernel_name,
        "isFusible": True,
    }


def _normalize_model_instance_id(value, default="mi01"):
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    match = re.match(r"^mi(\d+)$", text)
    if match:
        return f"mi{int(match.group(1)):02d}"
    return text


def _report_model_instance_id(report, default="mi01"):
    if not isinstance(report, dict):
        return default
    return _normalize_model_instance_id(report.get("model_instance_id"), default=default)


def _report_model_instance_index(report, default=1):
    if not isinstance(report, dict):
        return default
    return int(report.get("model_instance_index") or default)


def _report_model_instance_count(report, default=1):
    if not isinstance(report, dict):
        return default
    return int(report.get("model_instance_count") or default)


def _normalize_partition_mode(value):
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return text


def _stats_model_instance_partition_mode(stats, *, count):
    if not isinstance(stats, dict):
        return "single" if count <= 1 else "unknown"
    return _normalize_partition_mode(
        stats.get("model_instance_partition_mode") or ("single" if count <= 1 else "unknown")
    )


def _stats_model_instance_partition_verified(stats, *, count):
    if not isinstance(stats, dict):
        return count <= 1
    return bool(stats.get("model_instance_partition_verified", count <= 1))


def _resolve_model_instance_trace_context(input_path, model_instance_id):
    model_dir = find_model_dir(input_path)
    reports = collect_update_model_instance_reports(model_dir)
    model_instance_count = len(reports)
    requested_model_instance_id = (
        _normalize_model_instance_id(model_instance_id, default="") if model_instance_id else None
    )
    if model_instance_count > 1 and not requested_model_instance_id:
        raise ValueError(
            "multi-model-instance tracing requires --model-instance-id; refusing to build a whole-model trace"
        )

    selected_report = None
    if model_instance_count <= 1:
        selected_report = reports[0] if reports else None
        selected_model_instance_id = _report_model_instance_id(selected_report) if selected_report else "mi01"
        if requested_model_instance_id and requested_model_instance_id != selected_model_instance_id:
            raise ValueError(
                f"model-instance-id {requested_model_instance_id} does not exist for single-model-instance input"
            )
    else:
        for report in reports:
            if _report_model_instance_id(report) == requested_model_instance_id:
                selected_report = report
                break
        if selected_report is None:
            raise ValueError(
                f"model-instance-id {requested_model_instance_id} was not found in the current model input"
            )

    if not isinstance(selected_report, dict):
        raise ValueError("failed to resolve tracing inputs from model-instance reports")

    graph_library = selected_report.get("graph_library", {})
    node_library = graph_library.get("node_library", {}) if isinstance(graph_library, dict) else {}
    scope_library = selected_report.get("scope_library", {})
    fused_library = scope_library.get("fused_library", {}) if isinstance(scope_library, dict) else {}
    device_task_library = scope_library.get("device_task_library", {}) if isinstance(scope_library, dict) else {}
    stats = node_library.get("stats", {}) if isinstance(node_library, dict) else {}
    model_instance_partition_verified = _stats_model_instance_partition_verified(stats, count=model_instance_count)
    if model_instance_count > 1 and not model_instance_partition_verified:
        raise ValueError(
            "model-instance-local tracing inputs for "
            f"{_report_model_instance_id(selected_report)} are not safely partitioned"
        )

    node_rows = node_library.get("nodes", []) if isinstance(node_library, dict) else []
    nodes = [
        _node_from_library_row(row) for row in node_rows if isinstance(row, dict) and row.get("node_id") is not None
    ]
    return {
        "model_dir": model_dir,
        "nodes": nodes,
        "source_file": os.path.basename(str(node_library.get("path") or "")),
        "fused_functions": fused_library.get("functions", []) if isinstance(fused_library, dict) else [],
        "device_sections": device_task_library.get("sections", []) if isinstance(device_task_library, dict) else [],
        "task_identity_diagnostics": device_task_library.get("task_identity_diagnostics", {})
        if isinstance(device_task_library, dict)
        else {},
        "model_instance_scope_mode": "model_instance_id" if model_instance_count > 1 else "single",
        "model_instance_id": _report_model_instance_id(selected_report),
        "model_instance_index": _report_model_instance_index(selected_report),
        "model_instance_count": _report_model_instance_count(selected_report, default=model_instance_count or 1),
        "model_instance_partition_mode": _stats_model_instance_partition_mode(stats, count=model_instance_count),
        "model_instance_partition_verified": model_instance_partition_verified,
        "cross_report_index_scope_verified": True,
        "model_instance_partition_source": "model_instance_report",
    }


def build_event_edges(nodes):
    """
    根据 eventId 构建跨流连边:
    - EventNotify -> EventWait (相同 eventId)
    - MemoryWrite -> MemoryWait (相同 eventId)
    返回 edges 列表: [{"from": node_id, "to": node_id, "eventId": ..., "type": ...}]
    """
    notify_nodes = defaultdict(list)  # eventId -> [node, ...]
    wait_nodes = defaultdict(list)
    mw_nodes = defaultdict(list)  # eventId -> [node, ...] (MemoryWrite)
    mrw_nodes = defaultdict(list)  # eventId -> [node, ...] (MemoryWait)

    for n in nodes:
        eid = n.get("eventId")
        if not eid:
            continue
        dt = n.get("displayType", "")
        if dt == "EventNotify":
            notify_nodes[eid].append(n)
        elif dt == "EventWait":
            wait_nodes[eid].append(n)
        elif dt == "MemoryWrite":
            mw_nodes[eid].append(n)
        elif dt == "MemoryWait":
            mrw_nodes[eid].append(n)

    edges = []

    # EventNotify -> EventWait 连边
    for eid, notifies in notify_nodes.items():
        waits = wait_nodes.get(eid, [])
        for nf in notifies:
            for wt in waits:
                if nf["streamId"] != wt["streamId"]:
                    edges.append(
                        {
                            "from": nf["nodeId"],
                            "to": wt["nodeId"],
                            "eventId": eid,
                            "type": "event",
                            "label": f"Event {short_event_id(eid)}",
                        }
                    )

    # MemoryWrite -> MemoryWait 连边
    for eid, writes in mw_nodes.items():
        waits = mrw_nodes.get(eid, [])
        for mw in writes:
            for mrw in waits:
                if mw["streamId"] != mrw["streamId"]:
                    edges.append(
                        {
                            "from": mw["nodeId"],
                            "to": mrw["nodeId"],
                            "eventId": eid,
                            "type": "memory",
                            "label": f"Mem {short_event_id(eid)}",
                        }
                    )

    return edges


# ============================================================
# Chrome Tracing JSON 生成
# ============================================================


def build_trace_layout(nodes):
    """Build stable numeric pid/tid layout for tracing-compatible output."""
    stream_ids = sorted(set(n["streamId"] for n in nodes)) if nodes else []
    thread_ids = {sid: idx + 1 for idx, sid in enumerate(stream_ids)}
    edges_tid = len(stream_ids) + 1
    summary_tid = len(stream_ids) + 2
    return {
        "pid": TRACE_PROCESS_ID,
        "stream_ids": stream_ids,
        "thread_ids": thread_ids,
        "edges_tid": edges_tid,
        "summary_tid": summary_tid,
    }


def generate_tracing_json(nodes, edges, layout):
    """
    生成 Chrome Tracing 格式的 JSON 数据。

    设计思路:
    - 每个 streamId 占一个 track（泳道）
    - 节点在对应 stream 的 track 上按 nodeIdxInStream 排列
    - 使用 complete event 显示色块
    - 跨流连边使用 flow events 连接
    - ts 使用全局单调递增方案，保证 flow event 的 ts 约束
    """
    if not nodes:
        return [], [], {}

    # 按 streamId 分组，并按 nodeIdxInStream 排序
    streams = defaultdict(list)
    for n in nodes:
        streams[n["streamId"]].append(n)
    for sid in streams:
        streams[sid].sort(key=lambda x: x["nodeIdxInStream"])

    sorted_stream_ids = sorted(streams.keys())
    node_map = {n["nodeId"]: n for n in nodes}
    thread_ids = layout["thread_ids"]
    pid = layout["pid"]

    # === 全局 ts 分配 ===
    # Chrome Tracing 的 flow event 要求: 同一 id 的所有 step 的 ts 全局单调递增。
    # 策略: 所有 stream 共享同一个全局时间轴 (nodeIdxInStream)，但通过 tid 分泳道。
    # 这保证了同一 nodeIdxInStream 位置上的跨流节点 ts 一致，且 flow 的 from < to。
    max_nodes_in_stream = max(len(streams[sid]) for sid in sorted_stream_ids)
    slot = 100  # 每个 nodeIdxInStream 位置占 100us

    events = []

    # 为每个 stream 的每个节点创建 complete event
    for sid in sorted_stream_ids:
        tid = thread_ids[sid]

        for n in streams[sid]:
            ts = n["nodeIdxInStream"] * slot
            dt = n.get("displayType", "Unknown")
            color = NODE_TYPE_COLORS.get(dt, "#999999")
            cat = NODE_TYPE_CATEGORIES.get(dt, "other")

            # 构建节点名称
            if dt == "Kernel":
                kname = n.get("kernelName", "")
                short_name = short_kernel_name(kname)
                node_name = f"[{n['nodeId']}] {short_name}"
            else:
                eid_short = short_event_id(n.get("eventId", ""))
                node_name = f"[{n['nodeId']}] {dt}"
                if eid_short:
                    node_name += f" (ev:{eid_short})"

            # 构建详细 args
            args = {
                "nodeId": n["nodeId"],
                "streamId": sid,
                "streamIdxInGraph": n.get("streamIdxInGraph", -1),
                "nodeIdxInStream": n["nodeIdxInStream"],
                "type": dt,
                "isFusible": n.get("isFusible", True),
            }
            if n.get("eventId"):
                args["eventId"] = n["eventId"]
            if n.get("eventFlag"):
                args["eventFlag"] = n["eventFlag"]
            if n.get("kernelName"):
                args["kernelName"] = n["kernelName"]

            # 使用 complete event 展示节点（有宽度的色块）
            events.append(
                {
                    "ph": "X",  # Complete Event
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "dur": slot - 1,  # 几乎占满 slot
                    "name": node_name,
                    "cat": cat,
                    "color": color,
                    "args": args,
                }
            )

    # === 跨流连边: 双重可视化 ===
    # 1. Flow events (s/f): 在 Chrome Tracing 中显示为箭头连线
    #    - id 使用字符串格式 (更好的兼容性)
    #    - bp 字段绑定到切片边缘
    # 2. Edges track: 用 instant event 在专用轨道上标注每条连边
    #    - 作为 flow arrow 不可见时的备选方案
    flow_id = 0
    for edge in edges:
        src = node_map[edge["from"]]
        dst = node_map[edge["to"]]
        edge_color = "#f97316" if edge["type"] == "event" else "#a855f7"
        edge_label = edge["label"]

        # --- 时间戳策略 ---
        # Chrome Tracing flow event 约束:
        #   1. s.ts 必须 < f.ts (严格小于)
        #   2. ts 必须 >= 0
        #
        # 使用全局递增偏移来避免 ts 冲突:
        #   每个 edge 占用 slot 宽度, 从所有节点之后开始排列
        #   这样可以保证 ts 全局单调递增且不与节点重叠
        edge_start_ts = max_nodes_in_stream * slot + flow_id * slot
        s_ts = edge_start_ts
        f_ts = edge_start_ts + slot // 2

        flow_id_str = f"flow_{flow_id}"

        # Flow start: 在 source 节点所在的 track 上
        events.append(
            {
                "ph": "s",
                "pid": pid,
                "tid": thread_ids[src["streamId"]],
                "ts": s_ts,
                "id": flow_id_str,
                "cat": edge["type"],
                "name": edge_label,
                "bp": "e",  # bind to end of slice
                "args": {
                    "from_nodeId": src["nodeId"],
                    "to_nodeId": dst["nodeId"],
                    "eventId": edge["eventId"],
                    "edge_type": edge["type"],
                },
                "color": edge_color,
            }
        )

        # Flow finish: 在 dest 节点所在的 track 上
        events.append(
            {
                "ph": "f",
                "pid": pid,
                "tid": thread_ids[dst["streamId"]],
                "ts": f_ts,
                "id": flow_id_str,
                "cat": edge["type"],
                "name": edge_label,
                "bp": "s",  # bind to start of slice
                "args": {
                    "from_nodeId": src["nodeId"],
                    "to_nodeId": dst["nodeId"],
                    "eventId": edge["eventId"],
                    "edge_type": edge["type"],
                },
                "color": edge_color,
            }
        )

        # --- Edges track: instant event 标注连边 (备选可视化) ---
        edge_ts = edge_start_ts + slot // 4
        src_type = src.get("displayType", "?")
        dst_type = dst.get("displayType", "?")
        edge_name = f"{src_type}[{src['nodeId']}] -> {dst_type}[{dst['nodeId']}]  ({edge_label})"
        events.append(
            {
                "ph": "i",  # Instant event
                "pid": pid,
                "tid": layout["edges_tid"],
                "ts": edge_ts,
                "name": edge_name,
                "cat": f"edge_{edge['type']}",
                "s": "g",  # global instant
                "args": {
                    "from_nodeId": src["nodeId"],
                    "from_streamId": src["streamId"],
                    "from_type": src_type,
                    "to_nodeId": dst["nodeId"],
                    "to_streamId": dst["streamId"],
                    "to_type": dst_type,
                    "eventId": edge["eventId"],
                    "edge_type": edge["type"],
                },
                "color": edge_color,
            }
        )

        flow_id += 1

    return events


def generate_metadata(nodes, edges, layout):
    """生成元数据用于 metadata track"""
    stream_ids = sorted(set(n["streamId"] for n in nodes)) if nodes else []
    type_counts = defaultdict(int)
    for n in nodes:
        dt = n.get("displayType", "Unknown")
        type_counts[dt] += 1

    event_edge_count = sum(1 for e in edges if e["type"] == "event")
    memory_edge_count = sum(1 for e in edges if e["type"] == "memory")

    meta_events = []

    meta_events.append(
        {
            "ph": "M",
            "pid": layout["pid"],
            "name": "process_name",
            "args": {"name": "SuperKernel Graph"},
        }
    )

    meta_events.append(
        {
            "ph": "M",
            "pid": layout["pid"],
            "name": "process_sort_index",
            "args": {"sort_index": 0},
        }
    )

    stream_ids_sorted = sorted(stream_ids)
    for idx, sid in enumerate(stream_ids_sorted):
        meta_events.append(
            {
                "ph": "M",
                "pid": layout["pid"],
                "tid": layout["thread_ids"][sid],
                "name": "thread_name",
                "args": {"name": f"Stream {sid}"},
            }
        )
        meta_events.append(
            {
                "ph": "M",
                "pid": layout["pid"],
                "tid": layout["thread_ids"][sid],
                "name": "thread_sort_index",
                "args": {"sort_index": idx},
            }
        )

    meta_events.append(
        {
            "ph": "M",
            "pid": layout["pid"],
            "tid": layout["edges_tid"],
            "name": "thread_name",
            "args": {"name": "Edges (Notify↔Wait)"},
        }
    )
    meta_events.append(
        {
            "ph": "M",
            "pid": layout["pid"],
            "tid": layout["edges_tid"],
            "name": "thread_sort_index",
            "args": {"sort_index": len(stream_ids_sorted)},
        }
    )

    meta_events.append(
        {
            "ph": "M",
            "pid": layout["pid"],
            "tid": layout["summary_tid"],
            "name": "thread_name",
            "args": {"name": "Summary"},
        }
    )
    meta_events.append(
        {
            "ph": "M",
            "pid": layout["pid"],
            "tid": layout["summary_tid"],
            "name": "thread_sort_index",
            "args": {"sort_index": len(stream_ids_sorted) + 1},
        }
    )

    summary_text = (
        f"Total Nodes: {len(nodes)} | "
        f"Streams: {len(stream_ids)} ({','.join(str(s) for s in stream_ids)}) | "
        f"Event Edges: {event_edge_count} | "
        f"Memory Edges: {memory_edge_count} | " + " | ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))
    )
    meta_events.append(
        {
            "ph": "i",
            "pid": layout["pid"],
            "tid": layout["summary_tid"],
            "ts": 0,
            "name": summary_text,
            "cat": "metadata",
            "s": "g",
        }
    )

    return meta_events


def _task_resolved_graph_node_id(task, node_lookup_by_id, node_lookup_by_ordinal):
    if "graph_identity_valid" in task:
        if not task.get("graph_identity_valid"):
            return None
        resolved = task.get("resolved_graph_node_id")
        return int(resolved) if resolved is not None else None

    task_node_id = task.get("node_id", task.get("nodeId"))
    if task_node_id is not None:
        mapped = node_lookup_by_id.get(task_node_id)
        if mapped:
            resolved = mapped.get("node_id", mapped.get("nodeId"))
            return int(resolved) if resolved is not None else None
    task_index = task.get("task_index")
    if task_index is None:
        return None
    mapped = node_lookup_by_ordinal.get(task_index)
    if not mapped:
        return None
    resolved = mapped.get("node_id", mapped.get("nodeId"))
    return int(resolved) if resolved is not None else None


def build_cross_report_index(nodes, fused_functions, device_sections, task_identity_diagnostics=None):
    """构建 nodeId -> scope/sk/task 的跨报告索引。"""
    section_by_scope_name = {section["scope_name"]: section for section in device_sections if section.get("scope_name")}

    node_index = {}
    task_index_to_node_ids = defaultdict(list)
    for fused in fused_functions:
        section = section_by_scope_name.get(fused.get("scope_name"), {})
        fused_nodes = fused.get("node_details") or fused.get("nodes") or []
        node_lookup_by_ordinal = {idx: node for idx, node in enumerate(fused_nodes)}
        node_lookup_by_id = {
            node.get("node_id", node.get("nodeId")): node
            for node in fused_nodes
            if node.get("node_id", node.get("nodeId")) is not None
        }
        for node in fused_nodes:
            node_id = node.get("node_id", node.get("nodeId"))
            stream_id = node.get("stream_id", node.get("streamId"))
            if node_id is None:
                continue
            node_index[node_id] = {
                "scopeId": fused["scope_id"],
                "skId": fused["sk_id"],
                "streamId": stream_id,
                "taskIndices": [],
            }
        for queue_name in ("AIC", "AIV"):
            for task in section.get("queues", {}).get(queue_name, []):
                mapped_node_id = _task_resolved_graph_node_id(task, node_lookup_by_id, node_lookup_by_ordinal)
                task_index = task.get("task_index")
                if mapped_node_id is None or task_index is None:
                    continue
                mapped = node_lookup_by_id.get(mapped_node_id) or node_lookup_by_ordinal.get(task_index)
                mapped_stream_id = mapped.get("stream_id", mapped.get("streamId")) if mapped else None
                if mapped_node_id is None:
                    continue
                if mapped_node_id not in task_index_to_node_ids[task_index]:
                    task_index_to_node_ids[task_index].append(mapped_node_id)
                node_index.setdefault(
                    mapped_node_id,
                    {
                        "scopeId": fused["scope_id"],
                        "skId": fused["sk_id"],
                        "streamId": mapped_stream_id,
                        "taskIndices": [],
                    },
                )
                if task_index not in node_index[mapped_node_id]["taskIndices"]:
                    node_index[mapped_node_id]["taskIndices"].append(task_index)

    for node in nodes:
        node_index.setdefault(
            node["nodeId"],
            {
                "scopeId": None,
                "skId": None,
                "streamId": node["streamId"],
                "taskIndices": [],
            },
        )

    return {
        "node_index": {str(key): value for key, value in sorted(node_index.items())},
        "task_index_to_node_ids": {str(key): value for key, value in sorted(task_index_to_node_ids.items())},
        "task_identity_diagnostics": task_identity_diagnostics or {},
    }


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="SK Node Tracing Parser - 解析 SuperKernel 日志节点信息，生成 Chrome Tracing JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 解析目录下所有 log 文件
    python sk_node_tracing.py "$LOG_DIR"

    # 解析单个 log 文件
    python sk_node_tracing.py "$PLOG_FILE"

    # 指定输出文件
    python sk_node_tracing.py "$LOG_DIR" -o output.json

    # 多文件时只取第一个匹配
    python sk_node_tracing.py "$LOG_DIR" --first-only
        """,
    )
    parser.add_argument(
        "input",
        help="日志目录或日志文件路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出 JSON 文件路径 (默认: <input>_tracing.json)",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        default=False,
        help="目录模式下只解析第一个包含有效数据的日志文件",
    )
    parser.add_argument(
        "--model-instance-id",
        default=None,
        help="多 model instance 输入时显式选择 model instance，例如 mi01",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)

    if not os.path.exists(input_path):
        _emit(f"[ERROR] 路径不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    all_nodes = []
    source_file = None
    fused_functions = []
    device_sections = []
    trace_context = None
    try:
        trace_context = _resolve_model_instance_trace_context(input_path, args.model_instance_id)
    except FileNotFoundError:
        trace_context = None
    except ValueError as exc:
        _emit(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if trace_context is not None:
        all_nodes = trace_context["nodes"]
        source_file = trace_context["source_file"]
        fused_functions = trace_context["fused_functions"]
        device_sections = trace_context["device_sections"]
        _emit(
            "使用 model-instance-aware tracing 输入: "
            f"mode={trace_context['model_instance_scope_mode']}, "
            f"model_instance={trace_context['model_instance_id']}, "
            f"verified={trace_context['model_instance_partition_verified']}"
        )
    else:
        # 收集日志文件
        log_files = collect_log_files(input_path)
        if not log_files:
            _emit(f"[ERROR] 未找到日志文件: {input_path}", file=sys.stderr)
            sys.exit(1)

        _emit(f"找到 {len(log_files)} 个日志文件")

        # 解析所有文件
        for fpath in log_files:
            _emit(f"  解析: {os.path.basename(fpath)} ...", end=" ", flush=True)
            nodes = parse_log_file(fpath)
            if nodes:
                _emit(f"✓ 找到 {len(nodes)} 个节点")
                if not all_nodes:
                    all_nodes = nodes
                    source_file = fpath
                    if args.first_only:
                        break
            else:
                _emit("- 无 UpdateNodeScopeBitFlags 数据")

    if not all_nodes:
        _emit("[ERROR] 未从任何日志文件中提取到节点数据", file=sys.stderr)
        _emit(
            "  请确保日志中包含 UpdateNodeScopeBitFlags 的 Processed node 行，或提供 sk_node_detail.log",
            file=sys.stderr,
        )
        sys.exit(1)

    _emit("\n解析结果:")
    _emit(f"  来源文件: {os.path.basename(source_file) if source_file else ''}")
    _emit(f"  总节点数: {len(all_nodes)}")

    # 统计
    stream_ids = sorted(set(n["streamId"] for n in all_nodes))
    type_counts = defaultdict(int)
    for n in all_nodes:
        type_counts[n.get("displayType", "Unknown")] += 1
    _emit(f"  流 (Stream): {stream_ids} (共 {len(stream_ids)} 条)")
    for t, c in sorted(type_counts.items()):
        _emit(f"    {t}: {c}")

    # 构建跨流连边
    edges = build_event_edges(all_nodes)
    event_edges = [e for e in edges if e["type"] == "event"]
    memory_edges = [e for e in edges if e["type"] == "memory"]
    _emit(f"  跨流连边: 事件={len(event_edges)}, 内存={len(memory_edges)}, 共={len(edges)}")

    # 生成 Chrome/Edge Tracing JSON
    trace_layout = build_trace_layout(all_nodes)
    tracing_events = generate_tracing_json(all_nodes, edges, trace_layout)
    metadata_events = generate_metadata(all_nodes, edges, trace_layout)
    all_events = metadata_events + tracing_events
    cross_report_index = build_cross_report_index(
        all_nodes,
        fused_functions,
        device_sections,
        trace_context.get("task_identity_diagnostics", {}) if isinstance(trace_context, dict) else {},
    )
    model_instance_metadata = trace_context or {
        "model_instance_scope_mode": "single",
        "model_instance_id": "mi01",
        "model_instance_index": 1,
        "model_instance_count": 1,
        "model_instance_partition_mode": "single",
        "model_instance_partition_verified": True,
        "cross_report_index_scope_verified": False,
        "model_instance_partition_source": "legacy_log_scan",
    }

    tracing_data = {
        "traceEvents": all_events,
        "displayTimeUnit": "ns",
    }

    # 输出文件
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        if os.path.isdir(input_path):
            output_path = os.path.join(input_path, "sk_nodes_tracing.json")
        else:
            output_path = input_path.rsplit(".", 1)[0] + "_tracing.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tracing_data, f, indent=2, ensure_ascii=False)

    # 附加数据写入单独的 _meta.json 文件（不含 traceEvents，不影响 Chrome Tracing 解析）
    meta_output_path = output_path.rsplit(".", 1)[0] + "_meta.json"
    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "description": "SuperKernel ModelRI Graph Nodes",
                    "total_nodes": len(all_nodes),
                    "total_streams": len(stream_ids),
                    "stream_ids": stream_ids,
                    "event_edges": len(event_edges),
                    "memory_edges": len(memory_edges),
                    "type_counts": dict(type_counts),
                    "source": os.path.basename(source_file) if source_file else "",
                    "model_instance_scope_mode": model_instance_metadata["model_instance_scope_mode"],
                    "model_instance_id": model_instance_metadata["model_instance_id"],
                    "model_instance_index": model_instance_metadata["model_instance_index"],
                    "model_instance_count": model_instance_metadata["model_instance_count"],
                    "model_instance_partition_mode": model_instance_metadata["model_instance_partition_mode"],
                    "model_instance_partition_verified": model_instance_metadata["model_instance_partition_verified"],
                    "model_instance_partition_source": model_instance_metadata["model_instance_partition_source"],
                    "cross_report_index_scope_verified": model_instance_metadata["cross_report_index_scope_verified"],
                    "trace_compatibility": {
                        "viewer_targets": ["edge://tracing", "chrome://tracing"],
                        "pid_is_int": True,
                        "tid_is_int": True,
                        "display_time_unit": tracing_data["displayTimeUnit"],
                    },
                    "trace_layout": trace_layout,
                    "cross_report_keys": [
                        "nodeId",
                        "streamId",
                        "scopeId",
                        "skId",
                        "taskIndex",
                    ],
                },
                "cross_report_index": cross_report_index,
                "sk_nodes": all_nodes,
                "sk_edges": edges,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    file_size = os.path.getsize(output_path)
    _emit(f"\n✓ 输出文件: {output_path} ({file_size} bytes)")
    _emit(f"✓ 附加数据: {meta_output_path}")
    _emit("  用 Chrome 浏览器打开 chrome://tracing/ → 点击 'Load' → 加载此文件即可查看")


if __name__ == "__main__":
    main()
