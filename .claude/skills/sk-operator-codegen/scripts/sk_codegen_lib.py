# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Core codegen library for sk-operator-codegen.

Self-contained (no cross-skill imports). Provides:
- Signature parser for `__global__` kernel entries.
- SK kernel-type mapping (per kernel-launch-adapt.md s3.1 rule 2).
- SK adaptation renderer (Args struct + __sk__ template + SK_BIND).
- Template loader for templates/*.yaml.
- Remediation applier for the unified finding schema.
- SK form detector (none / current-sk-bind / partial / unknown).

Design constraint: codegen owns source mutation and adaptation; validation
rules are imported from sk-operator-validate so inline cleanup and the pipeline
use the same rule pack.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional


# ---------- Kernel-type mapping (kernel-launch-adapt.md s3.1 rule 2) ----------

_MIX_RE = re.compile(r"__mix__\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_ANY_MIX_RE = re.compile(r"__mix__\s*\((?P<body>[^)]*)\)")
_MIX_QUALIFIER_RE = r"__mix__(?:\s*\([^)]*\))?"


def _require_mapping_value(mapping: dict[Any, Any], key: Any, label: str) -> Any:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"{label} missing required key: {key!r}")
    return value


def map_kernel_type_for_sk(original_qualifiers: str) -> str:
    """Map an original __global__ function's kernel-type qualifier to the SK qualifier.

    Rules (from kernel-launch-adapt.md s3.1 rule 2):
      __vector__       -> __vector__
      __cube__         -> __cube__
      __mix__(c, v) general -> same __mix__(c, v)
      __mix__(1, 0)    -> __cube__   (special case)
      __mix__(0, 1)    -> __vector__ (special case)
    """
    text = original_qualifiers
    mix = _MIX_RE.search(text)
    if mix is not None:
        c, v = int(mix.group(1)), int(mix.group(2))
        if c >= 1 and v == 0:
            return "__cube__"
        if c == 0 and v >= 1:
            return "__vector__"
        return f"__mix__({c}, {v})"
    any_mix = _ANY_MIX_RE.search(text)
    if any_mix is not None:
        return (
            "__mix__("
            + ", ".join(part.strip() for part in any_mix.group("body").split(","))
            + ")"
        )
    if re.search(r"\b__mix__\b", text):
        return "__mix__"
    if "__vector__" in text:
        return "__vector__"
    if "__cube__" in text:
        return "__cube__"
    if "__aicore__" in text:
        return "__aicore__"
    raise ValueError(f"cannot map kernel type from qualifiers: {original_qualifiers!r}")


# ---------- Signature parser ----------

# Small integer C types that ABI requires alignas(4) for in Args structs.
SMALL_INT_TYPES = frozenset({"int8_t", "uint8_t", "int16_t", "uint16_t", "bool"})


@dataclass(frozen=True)
class ParsedParam:
    name: str
    c_type: str  # e.g. "GM_ADDR", "uint32_t", "int16_t"
    raw_source: str  # the raw "type name" snippet


@dataclass
class ParsedKernelEntry:
    name: str
    qualifiers_text: str  # e.g. "__global__ __vector__"
    return_type: str  # "void" (only supported)
    params: list[ParsedParam]
    body: str  # function body text (excluding braces)
    uses_get_block_num: bool
    template_params: list[dict[str, str]] = field(default_factory=list)


_GLOBAL_FN_RE = re.compile(
    r'(?P<extern>extern\s+"C"\s+)?'
    rf"(?P<qualifiers>(?:__global__|__aicore__|__vector__|__cube__|{_MIX_QUALIFIER_RE}|__inline__|inline|\s)+?)\s+"
    r"(?P<rettype>void)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"\((?P<params>[^()]*)\)\s*(?:\\\s*)?\{",
    re.MULTILINE,
)


def _find_matching_brace(text: str, open_pos: int) -> Optional[int]:
    if text[open_pos] != "{":
        raise ValueError("open_pos must point to an opening brace")
    depth = 0
    i = open_pos
    state = "code"
    quote = ""
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "line-comment":
            if ch == "\n":
                state = "code"
            i += 1
            continue
        if state == "block-comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            i += 1
            continue
        if state == "string":
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                state = "code"
                quote = ""
            i += 1
            continue
        if ch == "/" and nxt == "/":
            state = "line-comment"
            i += 2
            continue
        if ch == "/" and nxt == "*":
            state = "block-comment"
            i += 2
            continue
        if ch in {"'", '"'}:
            state = "string"
            quote = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _split_top_level_commas(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth_angle = 0
    depth_paren = 0
    for index, char in enumerate(text):
        if char == "<":
            depth_angle += 1
        elif char == ">" and depth_angle:
            depth_angle -= 1
        elif char == "(":
            depth_paren += 1
        elif char == ")" and depth_paren:
            depth_paren -= 1
        elif char == "," and depth_angle == 0 and depth_paren == 0:
            items.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def _template_params_before(source_text: str, match_start: int) -> list[dict[str, str]]:
    prefix = source_text[:match_start]
    match = re.search(r"template\s*<(?P<params>[^<>]+)>\s*$", prefix[-800:], re.DOTALL)
    if not match:
        return []
    template_params: list[dict[str, str]] = []
    for raw_part in _split_top_level_commas(match.group("params")):
        decl = raw_part.strip()
        if not decl:
            continue
        decl = decl.split("=", 1)[0].strip()
        name_match = re.search(r"(?:typename|class)\s+([A-Za-z_]\w*)\s*$", decl)
        if name_match is None:
            name_match = re.search(r"([A-Za-z_]\w*)\s*$", decl)
        if name_match is None:
            continue
        template_params.append({"decl": decl, "name": name_match.group(1)})
    return template_params


def _template_args_from_bind_target(bind_target: str, entry_name: str) -> list[str]:
    prefix = f"{entry_name}<"
    if not bind_target.startswith(prefix) or not bind_target.endswith(">"):
        return []
    return _split_top_level_commas(bind_target[len(prefix) : -1])


def _specialize_type(
    type_text: str, template_params: list[dict[str, str]], template_args: list[str]
) -> str:
    specialized = type_text
    for param, arg in zip(template_params, template_args, strict=False):
        specialized = re.sub(rf"\b{re.escape(param['name'])}\b", arg, specialized)
    return specialized


def _parse_param_list(params_text: str) -> list[ParsedParam]:
    text = params_text.strip()
    if not text:
        return []
    parts = _split_top_level_commas(text)
    parsed: list[ParsedParam] = []
    for part in parts:
        # Rightmost identifier is the param name; everything before is the type.
        m = re.search(r"([A-Za-z_]\w*)\s*$", part)
        if not m:
            raise ValueError(f"cannot parse parameter: {part!r}")
        name = m.group(1)
        name_start = m.start()
        c_type = part[:name_start].strip()
        if not c_type:
            raise ValueError(f"parameter missing type: {part!r}")
        parsed.append(ParsedParam(name=name, c_type=c_type, raw_source=part))
    return parsed


def _strip_macro_continuations(text: str) -> str:
    return re.sub(r"\\[ \t]*(?:\r?\n|$)", "\n", text)


def _macro_definition_context(source_text: str, pos: int) -> tuple[str, int] | None:
    line_start = source_text.rfind("\n", 0, pos) + 1
    scan_start = line_start
    while scan_start > 0:
        previous_end = scan_start - 1
        previous_start = source_text.rfind("\n", 0, previous_end) + 1
        previous_line = source_text[previous_start:previous_end]
        if not previous_line.rstrip().endswith("\\"):
            break
        scan_start = previous_start
    prefix = source_text[scan_start:pos]
    match = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)", prefix)
    if match is None:
        return None
    line_start = scan_start
    while True:
        line_end = source_text.find("\n", line_start)
        if line_end == -1:
            return match.group(1), len(source_text)
        line = source_text[line_start:line_end]
        if not line.rstrip().endswith("\\"):
            return match.group(1), line_end + 1
        line_start = line_end + 1


def _scan_parenthesized_call_end(source_text: str, open_paren: int) -> int | None:
    if source_text[open_paren] != "(":
        raise ValueError("open_paren must point to an opening parenthesis")
    depth = 0
    index = open_paren
    state = "code"
    quote = ""
    while index < len(source_text):
        char = source_text[index]
        nxt = source_text[index + 1] if index + 1 < len(source_text) else ""
        if state == "line-comment":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if char == "*" and nxt == "/":
                state = "code"
                index += 2
                continue
            index += 1
            continue
        if state == "string":
            if char == "\\":
                index += 2
                continue
            if char == quote:
                state = "code"
                quote = ""
            index += 1
            continue
        if char == "/" and nxt == "/":
            state = "line-comment"
            index += 2
            continue
        if char == "/" and nxt == "*":
            state = "block-comment"
            index += 2
            continue
        if char in {"'", '"'}:
            state = "string"
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _macro_invocation_end(source_text: str, start: int, macro_name: str) -> int:
    non_ws = re.search(r"\S", source_text[start:])
    if non_ws is None:
        return start
    pos = start + non_ws.start()
    match = re.match(rf"{re.escape(macro_name)}\s*\(", source_text[pos:])
    if match is None:
        return start
    open_paren = source_text.find("(", pos, pos + match.end())
    if open_paren == -1:
        return start
    end = _scan_parenthesized_call_end(source_text, open_paren)
    if end is None:
        return start
    while end < len(source_text) and source_text[end] in " \t;":
        end += 1
    if end < len(source_text) and source_text[end] == "\n":
        end += 1
    return end


def _adaptation_insertion_end(
    source_text: str, match: re.Match[str], close_brace: int
) -> int:
    context = _macro_definition_context(source_text, match.start())
    if context is None:
        return close_brace + 1
    macro_name, definition_end = context
    invocation_end = _macro_invocation_end(source_text, definition_end, macro_name)
    return invocation_end if invocation_end > definition_end else close_brace + 1


def parse_global_entries(source_text: str) -> list[ParsedKernelEntry]:
    """Parse all `__global__` entries from a source text."""
    entries: list[ParsedKernelEntry] = []
    for match in _GLOBAL_FN_RE.finditer(source_text):
        qualifiers = " ".join(match.group("qualifiers").split())
        if "__global__" not in qualifiers:
            continue
        open_brace = match.end() - 1
        close_brace = _find_matching_brace(source_text, open_brace)
        if close_brace is None:
            continue
        body_start = open_brace + 1
        body = _strip_macro_continuations(source_text[body_start:close_brace])
        try:
            params = _parse_param_list(match.group("params"))
        except ValueError:
            continue
        entries.append(
            ParsedKernelEntry(
                name=match.group("name"),
                qualifiers_text=qualifiers,
                return_type=match.group("rettype"),
                params=params,
                body=body,
                uses_get_block_num=bool(
                    re.search(r"\bAscendC\s*::\s*GetBlockNum\s*\(", body)
                ),
                template_params=_template_params_before(source_text, match.start()),
            )
        )
    return entries


# ---------- SK form detection ----------


@dataclass(frozen=True)
class SKFormAnalysis:
    form: str  # "none" / "current-sk-bind" / "partial" / "unknown"
    has_global: bool
    has_sk_keyword: bool
    has_sk_bind: bool
    has_unsupported_sk_signal: bool
    notes: list[str] = field(default_factory=list)


_UNSUPPORTED_SK_SIGNAL_RE = re.compile(
    r"__s"
    r"pk__\b|Fun"
    r"Level(?:MixCoreType|KType)\s+\w+\s+__attribute__|\."
    r"ascend\.meta"
)


def detect_sk_form(source_text: str) -> SKFormAnalysis:
    """Detect which SK adaptation form a source file uses.

    - none: only __global__ (clean input). Ready for adapt-sk-from-global.
    - current-sk-bind: __sk__ + SK_BIND template style (kernel-launch-adapt.md current form).
    - partial: mixed signals (e.g. has __sk__ but no SK_BIND).
    - unknown: no recognisable form (no __global__ at all -- not a kernel asset).
    """
    has_global = bool(re.search(r"__global__\b", source_text))
    has_sk = bool(re.search(r"__sk__\b", source_text))
    has_sk_bind = bool(re.search(r"\bSK_BIND\s*\(", source_text))
    has_unsupported_sk_signal = bool(_UNSUPPORTED_SK_SIGNAL_RE.search(source_text))

    notes: list[str] = []
    if not has_global:
        return SKFormAnalysis(
            form="unknown",
            has_global=False,
            has_sk_keyword=has_sk,
            has_sk_bind=has_sk_bind,
            has_unsupported_sk_signal=has_unsupported_sk_signal,
            notes=["no __global__ entry detected"],
        )
    is_current_sk_bind = has_sk and has_sk_bind and not has_unsupported_sk_signal
    if is_current_sk_bind:
        return SKFormAnalysis(
            form="current-sk-bind",
            has_global=True,
            has_sk_keyword=True,
            has_sk_bind=True,
            has_unsupported_sk_signal=False,
            notes=notes,
        )
    has_partial_sk_signal = has_sk or has_sk_bind or has_unsupported_sk_signal
    if has_partial_sk_signal:
        return SKFormAnalysis(
            form="partial",
            has_global=True,
            has_sk_keyword=has_sk,
            has_sk_bind=has_sk_bind,
            has_unsupported_sk_signal=has_unsupported_sk_signal,
            notes=[
                "source contains SK markers outside the supported current binding form"
            ],
        )
    return SKFormAnalysis(
        form="none",
        has_global=True,
        has_sk_keyword=False,
        has_sk_bind=False,
        has_unsupported_sk_signal=False,
        notes=[],
    )


def _human_finding(
    finding_id: str, message: str, evidence: list[str] | None = None
) -> dict:
    return {
        "finding_id": finding_id,
        "rule_id": finding_id,
        "severity": "blocker",
        "category": "codegen",
        "actionable_by": ["human"],
        "remediation_hint": {"kind": "human-decision"},
        "evidence": evidence or [],
        "message": message,
    }


# ---------- SK adaptation renderer ----------


def _camel_case(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)


def _is_small_int_type(c_type: str) -> bool:
    return any(t in c_type for t in SMALL_INT_TYPES)


def _type_references_template_params(
    c_type: str, template_param_names: list[str]
) -> bool:
    for name in template_param_names:
        if re.search(rf"\b{re.escape(name)}\b", c_type):
            return True
    return False


def _args_template_params_for_fields(
    params: list[ParsedParam], template_params: list[dict[str, str]]
) -> list[dict[str, str]]:
    referenced_template_params = []
    for template_param in template_params:
        for param in params:
            if _type_references_template_params(param.c_type, [template_param["name"]]):
                referenced_template_params.append(template_param)
                break
    return referenced_template_params


def _tpipe_declarations_with_depth(body: str) -> list[tuple[str, int]]:
    declarations: list[tuple[str, int]] = []
    depth = 0
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        for match in re.finditer(
            r"\b(?:AscendC::)?TPipe\s+([A-Za-z_]\w*)\s*(?:;|=)", line
        ):
            declarations.append((match.group(1), depth))
        depth += line.count("{") - line.count("}")
        if depth < 0:
            depth = 0
    seen: set[str] = set()
    unique: list[tuple[str, int]] = []
    for name, item_depth in declarations:
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, item_depth))
    return unique


def _tpipe_declarations(body: str) -> list[str]:
    return [name for name, _depth in _tpipe_declarations_with_depth(body)]


def _ensure_tpipe_destroy_without_pipe_all(body: str) -> str:
    lines = body.rstrip("\n").splitlines()
    if not lines:
        return body
    depths: list[int] = []
    depth = 0
    for line in lines:
        depths.append(depth)
        if line.strip().startswith("//"):
            continue
        depth += line.count("{") - line.count("}")
        if depth < 0:
            depth = 0

    insertions: dict[int, list[str]] = {}
    for index, line in enumerate(lines):
        if line.strip().startswith("//"):
            continue
        for match in re.finditer(
            r"\b(?:AscendC::)?TPipe\s+([A-Za-z_]\w*)\s*(?:;|=)", line
        ):
            name = match.group(1)
            close_index = None
            for probe in range(index + 1, len(lines)):
                if depths[probe] == depths[index] and lines[probe].strip().startswith(
                    "}"
                ):
                    close_index = probe
                    break
            if close_index is None:
                close_index = len(lines)
            scope_text = "\n".join(lines[index:close_index])
            if re.search(
                rf"\b{re.escape(name)}\s*\.\s*DestroyWithoutPipeAll\s*\(", scope_text
            ):
                continue
            indent_end = len(line) - len(line.lstrip())
            indent = line[:indent_end]
            insertions.setdefault(close_index, []).append(
                f"{indent}{name}.DestroyWithoutPipeAll();"
            )
    if not insertions:
        return body
    rendered: list[str] = []
    for index, line in enumerate(lines):
        rendered.extend(insertions.get(index, []))
        rendered.append(line)
    rendered.extend(insertions.get(len(lines), []))
    return "\n".join(rendered) + "\n"


def _strip_global_only_kernel_task_type_macros(body: str) -> str:
    """Drop global-kernel-only task type macros from generated __sk__ bodies."""
    rendered: list[str] = []
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            rendered.append(line)
            continue
        if re.fullmatch(r"KERNEL_TASK_TYPE_DEFAULT\s*\([^;]*\)\s*;?", stripped):
            continue
        cleaned = re.sub(r"\bKERNEL_TASK_TYPE_DEFAULT\s*\([^;]*\)\s*;?", "", line)
        if cleaned.strip() or line.endswith("\n"):
            rendered.append(cleaned)
    return "".join(rendered)


@dataclass
class RenderedSKAdaptation:
    sk_kernel_type: str
    args_struct_name: str
    args_struct_text: str
    sk_function_text: str
    sk_bind_text: str
    uses_sys_args: bool


def _cpp_identifier_from_symbol(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", text)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "op"
    if safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def render_sk_adaptation(
    entry: ParsedKernelEntry,
    *,
    mask: int = 4,
    num_splits: int = 4,
    sys_args_mode: str = "auto",  # "auto" | "always" | "never"
    body_override: str | None = None,
    sk_function_base: str | None = None,
    bind_target: str | None = None,
    unpack_args: bool = True,
    template_params: list[dict[str, str]] | None = None,
    template_args: list[str] | None = None,
) -> RenderedSKAdaptation:
    """Render the SK-adapted source fragments for one __global__ entry.

    Returns the pieces that get spliced into the source file:
    - Args struct declaration
    - __sk__ templated function body
    - SK_BIND macro line
    Plus metadata (sk_kernel_type, uses_sys_args).
    """
    if not (1 <= num_splits <= 4):
        raise ValueError("num_splits must be in 1..4 (SK_BIND limit)")
    if mask not in range(0, 8):
        raise ValueError("mask must be in 0..7 (DCCI/early-start bit mask)")
    template_params = template_params or []
    template_args = template_args or []
    if bool(template_params) != bool(template_args):
        raise ValueError("template_params and template_args must be provided together")
    if template_params and len(template_params) != len(template_args):
        raise ValueError("template parameter count must match template argument count")

    sk_kt = map_kernel_type_for_sk(entry.qualifiers_text)
    function_base = sk_function_base or f"{entry.name}_sk"
    args_name = f"{_camel_case(function_base.removesuffix('_sk'))}Args"
    bind_symbol = bind_target or entry.name
    template_param_decls = [param["decl"] for param in template_params]
    args_template_params = _args_template_params_for_fields(
        entry.params, template_params
    )
    args_template_decls = [param["decl"] for param in args_template_params]
    args_template_names = [param["name"] for param in args_template_params]
    args_type = (
        f"{args_name}<{', '.join(args_template_names)}>"
        if args_template_params
        else args_name
    )

    # ----- Args struct -----
    if entry.params:
        args_lines = []
        if args_template_params:
            args_lines.append(f"template<{', '.join(args_template_decls)}>")
        args_lines.append(f"struct {args_name} " + "{")
        for p in entry.params:
            align_prefix = "alignas(4) " if _is_small_int_type(p.c_type) else ""
            args_lines.append(f"    {align_prefix}{p.c_type} {p.name};")
        args_lines.append("};")
        args_struct = "\n".join(args_lines)
    else:
        args_struct = ""

    # ----- sysArgs decision -----
    if sys_args_mode == "always":
        use_sys = True
    elif sys_args_mode == "never":
        use_sys = False
    elif sys_args_mode == "auto":
        use_sys = entry.uses_get_block_num
    else:
        raise ValueError(
            f"sys_args_mode must be auto|always|never, got {sys_args_mode!r}"
        )

    sig_args = f"const {args_type} *args" if entry.params else ""
    if use_sys:
        sig_args += (", " if sig_args else "") + "sk::SkSystemArgs *sysArgs"

    # ----- SK function body -----
    unpack_lines = []
    if unpack_args:
        for p in entry.params:
            unpack_lines.append(f"    {p.c_type} {p.name} = args->{p.name};")

    # Rewrite AscendC::GetBlockNum() -> sysArgs->skNumBlocks if sysArgs is in scope.
    rewritten_body = body_override if body_override is not None else entry.body
    if use_sys:
        rewritten_body = re.sub(
            r"\bAscendC\s*::\s*GetBlockNum\s*\(\s*\)",
            "sysArgs->skNumBlocks",
            rewritten_body,
        )
    rewritten_body = _strip_global_only_kernel_task_type_macros(rewritten_body)
    rewritten_body = _ensure_tpipe_destroy_without_pipe_all(rewritten_body)

    sk_func = (
        f"template<{', '.join(template_param_decls + ['uint32_t splitidx'])}>\n"
        f"__sk__ {sk_kt} void {function_base}({sig_args})\n"
        f"{{\n"
        + "\n".join(unpack_lines)
        + ("\n\n" if unpack_lines else "")
        + rewritten_body.rstrip("\n")
        + "\n}\n"
    )

    # ----- SK_BIND -----
    split_args = ", ".join(
        f"{function_base}<{', '.join(template_args + [str(i)])}>"
        if template_args
        else f"{function_base}<{i}>"
        for i in range(num_splits)
    )
    sk_bind = f"SK_BIND({bind_symbol}, {mask}, {split_args});"

    return RenderedSKAdaptation(
        sk_kernel_type=sk_kt,
        args_struct_name=args_name,
        args_struct_text=args_struct,
        sk_function_text=sk_func,
        sk_bind_text=sk_bind,
        uses_sys_args=use_sys,
    )


# ---------- Source file mutation ----------

_KERNEL_OPERATOR_INCLUDE = '#include "kernel_operator.h"'


def _ensure_kernel_operator_include(source_text: str) -> str:
    if _KERNEL_OPERATOR_INCLUDE in source_text:
        return source_text
    lines = source_text.splitlines(keepends=True)
    insert_at = 0
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#include"):
            insert_at = index + 1
    lines.insert(insert_at, _KERNEL_OPERATOR_INCLUDE + "\n")
    return "".join(lines)


def adapt_source_text(
    source_text: str,
    *,
    mask: int = 4,
    num_splits: int = 4,
    sys_args_mode: str = "auto",
    bind_targets_by_entry: dict[str, str] | None = None,
) -> tuple[str, list[dict]]:
    """Adapt every __global__ entry in a source text.

    Inserts:
    - At the top (after the last include): no-op (existing includes preserved).
    - Below each __global__ function's closing brace: args struct + sk function + SK_BIND.

    Returns (new_source_text, [per_entry_meta...]).
    """
    entries = parse_global_entries(source_text)
    if not entries:
        raise ValueError("no __global__ entry found; nothing to adapt")

    # Locate end of each entry to splice after.
    # We rebuild the file by walking through entries in source order.
    pieces: list[str] = []
    last_end = 0
    metas: list[dict] = []
    bind_targets_by_entry = bind_targets_by_entry or {}
    for entry in entries:
        m = next(
            x
            for x in _GLOBAL_FN_RE.finditer(source_text)
            if x.group("name") == entry.name
        )
        open_brace = m.end() - 1
        close_brace = _find_matching_brace(source_text, open_brace)
        if close_brace is None:
            continue
        insertion_end = _adaptation_insertion_end(source_text, m, close_brace)
        # Append text up to the point where the original entry is visible to SK_BIND.
        pieces.append(source_text[last_end:insertion_end])
        bind_target = bind_targets_by_entry.get(entry.name, "")
        template_args = (
            _template_args_from_bind_target(bind_target, entry.name)
            if bind_target
            else []
        )
        template_params = entry.template_params if template_args else []
        rendered = render_sk_adaptation(
            entry,
            mask=mask,
            num_splits=num_splits,
            sys_args_mode=sys_args_mode,
            bind_target=bind_target or None,
            template_params=template_params,
            template_args=template_args,
        )
        pieces.append(
            "\n\n// ---- SK adaptation (auto-generated) ----\n"
            + (rendered.args_struct_text + "\n\n" if rendered.args_struct_text else "")
            + rendered.sk_function_text
            + "\n"
            + rendered.sk_bind_text
            + "\n// ---- end SK adaptation ----\n"
        )
        last_end = insertion_end
        parameters = []
        for p in entry.params:
            c_type = _specialize_type(p.c_type, template_params, template_args)
            parameters.append({"name": p.name, "c_type": c_type})
        resolved_bind_target = bind_target or entry.name
        metas.append(
            {
                "entry_name": entry.name,
                "sk_kernel_type": rendered.sk_kernel_type,
                "args_struct_name": rendered.args_struct_name,
                "uses_sys_args": rendered.uses_sys_args,
                "param_count": len(entry.params),
                "parameters": parameters,
                "original_qualifiers": entry.qualifiers_text,
                "bind_target": resolved_bind_target,
                "global_launch_target": resolved_bind_target,
            }
        )
    pieces.append(source_text[last_end:])
    return _ensure_kernel_operator_include("".join(pieces)), metas


# ---------- aclgraph-canonical pybind/package renderers ----------

_ACLGRAPH_REQUIRED_INCLUDES = [
    "#include <pybind11/pybind11.h>",
    "#include <torch/extension.h>",
    "#include <cstdint>",
    "#include <limits>",
    "#include <vector>",
    '#include "third_party/acl/inc/acl/acl_rt.h"',
    '#include "torch_npu/csrc/core/npu/NPUStream.h"',
    '#include "kernel_operator.h"',
]


def _ensure_aclgraph_includes(source_text: str) -> str:
    missing = [line for line in _ACLGRAPH_REQUIRED_INCLUDES if line not in source_text]
    if not missing:
        return source_text
    return "\n".join(missing) + "\n\n" + source_text


def _is_tensor_like_param(param: dict[str, Any]) -> bool:
    c_type = str(param.get("c_type", ""))
    return "GM_ADDR" in c_type or "__gm__" in c_type or "*" in c_type


def _is_scalar_param(param: dict[str, Any]) -> bool:
    c_type = str(param.get("c_type", ""))
    scalar_markers = (
        "int",
        "uint",
        "float",
        "double",
        "half",
        "bool",
        "size_t",
    )
    has_scalar_marker = False
    for marker in scalar_markers:
        if marker in c_type:
            has_scalar_marker = True
            break
    return not _is_tensor_like_param(param) and has_scalar_marker


def _runtime_param_kind(
    param: dict[str, Any], entry: dict[str, Any] | None = None
) -> str:
    io_params = {}
    if entry is not None:
        io_params = dict(entry.get("io_contract", {}).get("parameters", {}) or {})
    declared = io_params.get(str(param.get("name") or ""))
    declared_kind = (
        str(declared.get("kind") or "") if isinstance(declared, dict) else ""
    )
    if declared_kind and (_is_tensor_like_param(param) or not _is_scalar_param(param)):
        return declared_kind
    if _is_tensor_like_param(param):
        return "tensor"
    if _is_scalar_param(param):
        return "scalar"
    return "host_struct"


def _runtime_param_contract(
    param: dict[str, Any], entry: dict[str, Any] | None = None
) -> dict[str, Any]:
    if entry is None:
        return {}
    declared = dict(entry.get("io_contract", {}).get("parameters", {}) or {}).get(
        str(param.get("name") or "")
    )
    return declared if isinstance(declared, dict) else {}


def _is_nullable_runtime_param(
    param: dict[str, Any], entry: dict[str, Any] | None = None
) -> bool:
    return bool(_runtime_param_contract(param, entry).get("nullable"))


def _runtime_wrapper_contract(entry: dict[str, Any] | None = None) -> dict[str, Any]:
    if entry is None:
        return {}
    wrapper = entry.get("runtime_wrapper") or entry.get("io_contract", {}).get(
        "runtime_wrapper", {}
    )
    return wrapper if isinstance(wrapper, dict) else {}


def _has_runtime_wrapper(entry: dict[str, Any] | None = None) -> bool:
    wrapper = _runtime_wrapper_contract(entry)
    return bool(str(wrapper.get("entry") or "").strip())


def _runtime_wrapper_symbol(entry: dict[str, Any]) -> str:
    wrapper_entry = str(_runtime_wrapper_contract(entry).get("entry") or "").strip()
    return wrapper_entry or f"run_{entry['entry_name']}"


def _tensor_pointer_type(param: dict[str, Any]) -> str:
    c_type = str(param.get("c_type", "GM_ADDR"))
    if "GM_ADDR" in c_type:
        return "uint8_t *"
    return re.sub(r"\b__gm__\b", "", c_type).strip()


def _run_signature_params(entry: dict[str, Any]) -> list[str]:
    params = entry.get("parameters", [])
    if not params:
        return ["const at::Tensor &x"]
    rendered: list[str] = []
    for param in params:
        kind = _runtime_param_kind(param, entry)
        if kind == "tensor":
            if _is_nullable_runtime_param(param, entry):
                rendered.append(f"const pybind11::object &{param['name']}")
            else:
                rendered.append(f"const at::Tensor &{param['name']}")
        elif kind == "scalar":
            rendered.append(f"{param['c_type']} {param['name']}")
        elif kind == "host_struct":
            rendered.append(f"const std::vector<int64_t> &{param['name']}")
        elif kind == "tensor_list":
            if not _has_runtime_wrapper(entry):
                raise ValueError(
                    f"tensor_list parameter {param['name']} for {entry.get('entry_name', '<unknown>')} "
                    "requires a user/adapter-provided runtime wrapper"
                )
            rendered.append(f"const at::Tensor &{param['name']}")
        else:
            raise ValueError(
                f"unsupported runtime parameter kind {kind!r} for {param['name']}"
            )
    return rendered


def _launch_args(entry: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    for param in entry.get("parameters", []):
        kind = _runtime_param_kind(param, entry)
        if kind == "tensor":
            pointer_type = _tensor_pointer_type(param)
            if _is_nullable_runtime_param(param, entry):
                rendered.append(f"{param['name']}_ptr")
            else:
                rendered.append(f"({pointer_type})({param['name']}.mutable_data_ptr())")
        elif kind == "scalar":
            rendered.append(param["name"])
        elif kind == "host_struct":
            rendered.append(f"{param['name']}_host")
        elif kind == "tensor_list":
            raise ValueError(
                f"tensor_list parameter {param['name']} for {entry.get('entry_name', '<unknown>')} "
                "requires a user/adapter-provided runtime wrapper"
            )
        else:
            raise ValueError(
                f"unsupported runtime parameter kind {kind!r} for {param['name']}"
            )
    return rendered


def _runtime_setup_lines(entry: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for param in entry.get("parameters", []):
        kind = _runtime_param_kind(param, entry)
        if kind == "tensor" and _is_nullable_runtime_param(param, entry):
            name = str(param["name"])
            pointer_type = _tensor_pointer_type(param)
            lines.extend(
                [
                    f"    {pointer_type} {name}_ptr = nullptr;",
                    f"    at::Tensor {name}_tensor;",
                    f"    if (!{name}.is_none()) {{",
                    f"        {name}_tensor = pybind11::cast<at::Tensor>({name});",
                    f"        {name}_ptr = ({pointer_type})({name}_tensor.mutable_data_ptr());",
                    "    }",
                ]
            )
            continue
        if kind != "host_struct":
            continue
        name = str(param["name"])
        c_type = str(param["c_type"])
        lines.extend(
            [
                f"    TORCH_CHECK({name}.size() == static_cast<int64_t>(sizeof({c_type})),",
                (
                    f'                "host-struct parameter {name} expects ", '
                    f'sizeof({c_type}), " bytes, got ", {name}.size());'
                ),
                f"    {c_type} {name}_host{{}};",
                f"    auto *{name}_bytes = reinterpret_cast<uint8_t *>(&{name}_host);",
                f"    for (size_t i = 0; i < sizeof({c_type}); ++i) {{",
                f"        auto value = {name}[static_cast<int64_t>(i)];",
                (
                    f"        TORCH_CHECK(value >= 0 && value <= 255, "
                    f'"host-struct parameter {name} byte out of range at index ", i);'
                ),
                f"        {name}_bytes[i] = static_cast<uint8_t>(value);",
                "    }",
            ]
        )
    return lines


def _tensor_param_names(entry: dict[str, Any]) -> list[str]:
    params = entry.get("parameters", [])
    names: list[str] = []
    for param in params:
        kind = _runtime_param_kind(param, entry)
        if kind == "tensor" or (kind == "tensor_list" and _has_runtime_wrapper(entry)):
            names.append(param["name"])
    return names


def _first_tensor_param(entry: dict[str, Any]) -> str:
    names = _tensor_param_names(entry)
    return names[0] if names else "x"


def _return_tensor_param(entry: dict[str, Any]) -> str:
    explicit_return = str(entry.get("pybind_return_tensor") or "").strip()
    tensor_names = _tensor_param_names(entry)
    if explicit_return:
        if explicit_return not in tensor_names:
            raise ValueError(
                f"pybind_return_tensor {explicit_return!r} is not a tensor parameter for "
                f"{entry.get('entry_name', '<unknown>')}; tensor parameters: {tensor_names}"
            )
        return explicit_return
    return tensor_names[0] if tensor_names else "x"


def _torch_schema(entry: dict[str, Any]) -> str:
    params = entry.get("parameters", [])
    if not params:
        return f"{entry['entry_name']}(Tensor x) -> Tensor"
    rendered: list[str] = []
    for param in params:
        kind = _runtime_param_kind(param, entry)
        if kind == "tensor":
            rendered.append(
                f"Tensor? {param['name']}"
                if _is_nullable_runtime_param(param, entry)
                else f"Tensor {param['name']}"
            )
        elif kind == "host_struct":
            rendered.append(f"int[] {param['name']}")
        elif kind == "tensor_list":
            if not _has_runtime_wrapper(entry):
                raise ValueError(
                    f"tensor_list parameter {param['name']} for {entry.get('entry_name', '<unknown>')} "
                    "requires a user/adapter-provided runtime wrapper"
                )
            rendered.append(f"Tensor {param['name']}")
        elif "float" in param["c_type"] or "double" in param["c_type"]:
            rendered.append(f"float {param['name']}")
        elif "bool" in param["c_type"]:
            rendered.append(f"bool {param['name']}")
        else:
            rendered.append(f"int {param['name']}")
    return f"{entry['entry_name']}({', '.join(rendered)}) -> Tensor"


def _run_param_names(entry: dict[str, Any]) -> list[str]:
    params = entry.get("parameters", [])
    if not params:
        return ["x"]
    return [str(param["name"]) for param in params]


def _render_get_current_aicore_num() -> str:
    return """static uint32_t get_current_aicore_num()
{
    int32_t device_id = 0;
    auto ret = aclrtGetDevice(&device_id);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "Failed to get current NPU device id, acl error: ", ret);

    int64_t aicore_num = 0;
    ret = aclrtGetDeviceInfo(static_cast<uint32_t>(device_id), ACL_DEV_ATTR_AICORE_CORE_NUM, &aicore_num);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "Failed to get NPU AI Core num for device ", device_id, ", acl error: ", ret);
    TORCH_CHECK(aicore_num > 0 && aicore_num <= std::numeric_limits<uint32_t>::max(),
                "Invalid NPU AI Core num: ", aicore_num);
    return static_cast<uint32_t>(aicore_num);
}
"""


def _entry_block_dim_line(entry: dict[str, Any]) -> str:
    launch = entry.get("io_contract", {}).get("launch", {})
    if isinstance(launch, dict) and launch.get("block_dim") is not None:
        return f"    uint32_t blockDim = {int(launch['block_dim'])};"
    return "    uint32_t blockDim = ascendc_ops::get_current_aicore_num();"


def render_aclgraph_kernel_source(
    source_text: str, entries: list[dict[str, Any]]
) -> str:
    """Append aclgraph-style run_<op> wrappers to an adapted SK source."""
    lines = [
        _ensure_aclgraph_includes(source_text).rstrip(),
        "",
        "namespace ascendc_ops {",
        _render_get_current_aicore_num().rstrip(),
        "",
    ]
    for entry in entries:
        if _has_runtime_wrapper(entry):
            continue
        name = entry["entry_name"]
        signature = ", ".join(_run_signature_params(entry))
        sk_launch_target = entry.get("bind_target") or name
        launch_args = ", ".join(_launch_args(entry))
        return_tensor = _return_tensor_param(entry)
        lines.append(f"at::Tensor run_{name}({signature})")
        lines.append("{")
        lines.append(
            "    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);"
        )
        lines.append(_entry_block_dim_line(entry))
        lines.extend(_runtime_setup_lines(entry))
        lines.append(
            f"    {sk_launch_target}<<<blockDim, nullptr, acl_stream>>>({launch_args});"
        )
        lines.append(f"    return {return_tensor};")
        lines.append("}")
        lines.append("")
    lines.append("}  // namespace ascendc_ops")
    return "\n".join(lines).rstrip() + "\n"


def render_pybind11_asc(module_name: str, entries: list[dict[str, Any]]) -> str:
    lines = [
        "#include <pybind11/pybind11.h>",
        "#include <torch/extension.h>",
        "",
        "namespace ascendc_ops {",
    ]
    for entry in entries:
        signature = ", ".join(_run_signature_params(entry))
        lines.append(f"at::Tensor {_runtime_wrapper_symbol(entry)}({signature});")
    lines.extend(
        [
            "}",
            "",
            "#ifndef TORCH_EXTENSION_NAME",
            f"#define TORCH_EXTENSION_NAME {module_name}",
            "#endif",
            "",
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)",
            "{",
        ]
    )
    for entry in entries:
        name = entry["entry_name"]
        lines.append(
            f'    m.def("run_{name}", &ascendc_ops::{_runtime_wrapper_symbol(entry)}, "AscendC SK operator {name}");'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _is_aclgraph_pybind_source_name(name: str) -> bool:
    return name == "pybind11.asc" or (
        name.startswith("pybind11_") and name.endswith(".asc")
    )


def _aclgraph_entry_module_base(entry: dict[str, Any]) -> str:
    return _safe_identifier(str(entry["entry_name"]), "entry")


def _aclgraph_entry_pybind_name(entry: dict[str, Any]) -> str:
    return f"pybind11_{_aclgraph_entry_module_base(entry)}.asc"


def _aclgraph_csrc_rel(path_text: str) -> str:
    rel = PurePosixPath(str(path_text).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"invalid aclgraph csrc relative path: {path_text!r}")
    if rel.parts and rel.parts[0] == "csrc":
        rel = PurePosixPath(*rel.parts[1:])
    if not rel.parts:
        raise ValueError("empty aclgraph csrc relative path")
    return rel.as_posix()


def _normalize_arch_list(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    if isinstance(values, str):
        iterable = re.split(r"[,;]", values)
    else:
        iterable = values or []
    for item in iterable:
        arch = str(item or "").strip().lower().replace("_", "-")
        if not arch or arch in seen:
            continue
        seen.add(arch)
        result.append(arch)
    return result


def _aclgraph_entry_specs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    module_base_by_entry: dict[str, str] = {}
    entry_by_module_base: dict[str, str] = {}
    for entry in entries:
        entry_name = str(entry["entry_name"])
        module_base = _aclgraph_entry_module_base(entry)
        previous_entry = entry_by_module_base.get(module_base)
        if previous_entry is not None and previous_entry != entry_name:
            raise ValueError(
                "aclgraph entry names collide after module-name sanitization: "
                f"{previous_entry!r} and {entry_name!r} -> {module_base!r}"
            )
        entry_by_module_base[module_base] = entry_name
        module_base_by_entry[entry_name] = module_base
        csrc_file = str(entry.get("csrc_file") or f"{entry_name}.asc")
        wrapper_sources = []
        wrapper = _runtime_wrapper_contract(entry)
        wrapper_csrc_file = str(wrapper.get("csrc_file") or "").strip()
        if wrapper_csrc_file:
            wrapper_sources.append(f"csrc/{_aclgraph_csrc_rel(wrapper_csrc_file)}")
        specs.append(
            {
                "entry_name": entry_name,
                "module_base": module_base,
                "csrc_file": f"csrc/{_aclgraph_csrc_rel(csrc_file)}",
                "pybind_file": f"csrc/{_aclgraph_entry_pybind_name(entry)}",
                "extra_sources": wrapper_sources,
                "supported_arches": _normalize_arch_list(entry.get("supported_arches")),
                "supported_soc_versions": list(entry.get("supported_soc_versions", [])),
                "target_resolution": dict(entry.get("target_resolution", {})),
                "support_source": entry.get("support_source", ""),
                "compile_defines": list(
                    dict.fromkeys(entry.get("compile", {}).get("defines", []))
                ),
                "compile_options": list(
                    dict.fromkeys(entry.get("compile", {}).get("options", []))
                ),
            }
        )
    if len(module_base_by_entry) != len(entries):
        raise ValueError("duplicate aclgraph entry names are not supported")
    return specs


def render_pybind11_entry_asc(entry: dict[str, Any]) -> str:
    return render_pybind11_asc(_aclgraph_entry_module_base(entry), [entry])


def render_arch_selector_py(module_name: str, entries: list[dict[str, Any]]) -> str:
    entry_modules = {
        spec["entry_name"]: spec["module_base"]
        for spec in _aclgraph_entry_specs(entries)
    }
    entry_supported_arches = {
        spec["entry_name"]: spec.get("supported_arches", [])
        for spec in _aclgraph_entry_specs(entries)
    }
    template = '''"""Runtime NPU arch selection for ACLGraph custom ops."""
from __future__ import annotations

import importlib
import importlib.machinery
import os
from pathlib import Path

MODULE_BASE = "__MODULE_NAME__"
ENTRY_MODULES = __ENTRY_MODULES__
ENTRY_SUPPORTED_ARCHES = __ENTRY_SUPPORTED_ARCHES__
ENTRY_MODULE_PREFIXES = sorted(ENTRY_MODULES.items(), key=lambda item: len(item[1]), reverse=True)

# Source-backed automatic mappings only.
OFFICIAL_AUTO_SOC_ARCH_RULES = (
    {
        "arch": "dav-2201",
        "string_prefixes": ("ascend910b",),
        "enum_values": (104, 220, 221, 222, 223, 224, 225),
    },
    {
        "arch": "dav-3510",
        "string_prefixes": ("ascend950",),
        "enum_values": (260,),
    },
)


def normalize_arch(arch: object) -> str:
    return str(arch or "").strip().lower().replace("_", "-")


def arch_to_module_suffix(arch: object) -> str:
    return normalize_arch(arch).replace("-", "_")


def module_suffix_to_arch(module_suffix: str) -> str:
    if module_suffix.startswith("dav_"):
        return module_suffix.replace("_", "-")
    return module_suffix.replace("_", "-")


def module_name_for_entry_arch(entry_name: str, arch: object) -> str:
    return ENTRY_MODULES[entry_name] + "_" + arch_to_module_suffix(arch)


def _iter_extension_module_names(package_dir: Path):
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for path in package_dir.iterdir():
        if not path.is_file():
            continue
        for suffix in suffixes:
            if path.name.endswith(suffix):
                yield path.name[: -len(suffix)]
                break


def _discover_modules_by_arch(package_dir: Path) -> dict[str, dict[str, str]]:
    discovered: dict[str, dict[str, str]] = {}
    for found_module_name in _iter_extension_module_names(package_dir):
        for entry_name, module_base in ENTRY_MODULE_PREFIXES:
            prefix = module_base + "_"
            if not found_module_name.startswith(prefix):
                continue
            module_suffix = found_module_name[len(prefix) :]
            arch = module_suffix_to_arch(module_suffix)
            discovered.setdefault(arch, {})[entry_name] = found_module_name
            break
    return discovered


def discover_available_arches(package_dir=None) -> list[str]:
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    discovered = _discover_modules_by_arch(root)
    arches = [
        arch
        for arch, modules_by_entry in discovered.items()
        if modules_by_entry
    ]
    return sorted(arches)


def supported_arches_for_entry(entry_name: str) -> list[str]:
    declared = [normalize_arch(arch) for arch in ENTRY_SUPPORTED_ARCHES.get(entry_name, []) if normalize_arch(arch)]
    return sorted(dict.fromkeys(declared))


def _detect_soc_version():
    try:
        import torch_npu
    except Exception:
        return None
    npu = getattr(torch_npu, "npu", None)
    getter = getattr(npu, "get_soc_version", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def map_soc_version_to_arch(soc_version: object) -> str | None:
    if soc_version is None:
        return None
    raw = str(soc_version).strip()
    normalized = raw.lower().replace(" ", "").replace("_", "")
    enum_value = None
    try:
        enum_value = int(raw)
    except (TypeError, ValueError):
        enum_value = None

    for rule in OFFICIAL_AUTO_SOC_ARCH_RULES:
        arch = str(rule["arch"])
        for prefix in rule.get("string_prefixes", ()):
            if normalized.startswith(str(prefix).lower().replace("_", "")):
                return arch
        enum_values = rule.get("enum_values", ())
        if enum_value is not None and enum_value in {int(value) for value in enum_values}:
            return arch
    return None


def _format_arches(arches: list[str]) -> str:
    return ", ".join(arches) if arches else "<none>"


def select_arch(
    available_arches: list[str] | None = None,
    *,
    get_soc_version=None,
) -> str:
    raw_available = discover_available_arches() if available_arches is None else available_arches
    available = [normalize_arch(arch) for arch in raw_available if normalize_arch(arch)]
    available = sorted(dict.fromkeys(available))

    explicit_source = "SK_ACLGRAPH_NPU_ARCH"
    explicit = os.getenv(explicit_source)
    if explicit:
        requested = normalize_arch(explicit)
        if requested in available:
            return requested
        raise RuntimeError(
            "Requested %s=%r is not packaged. Available arches: %s."
            % (explicit_source, requested, _format_arches(available))
        )

    detector = get_soc_version or _detect_soc_version
    detected_soc = detector()
    detected_arch = map_soc_version_to_arch(detected_soc)
    if detected_arch is not None:
        if detected_arch in available:
            return detected_arch
        raise RuntimeError(
            "Detected SoC version %r maps to %s, but that arch is not packaged. Available arches: %s. "
            "Set SK_ACLGRAPH_NPU_ARCH explicitly to override when you know the target arch."
            % (detected_soc, detected_arch, _format_arches(available))
        )

    raise RuntimeError(
        "Unable to select ACLGraph custom ops NPU arch. Detected SoC version: %r. Available arches: %s. "
        "Set SK_ACLGRAPH_NPU_ARCH explicitly. Automatic selection only uses source-backed SoC mappings."
        % (detected_soc, _format_arches(available))
    )


def load_custom_ops_libs(
    *,
    package: str | None = None,
    available_arches: list[str] | None = None,
    get_soc_version=None,
):
    selected_arch = select_arch(available_arches, get_soc_version=get_soc_version)
    package_name = package or __package__
    discovered = _discover_modules_by_arch(Path(__file__).resolve().parent)
    modules_for_arch = discovered.get(selected_arch, {})
    modules_by_entry = {}
    for entry_name in ENTRY_MODULES:
        selected_module_name = modules_for_arch.get(entry_name)
        if not selected_module_name:
            continue
        module = importlib.import_module("." + selected_module_name, package_name)
        module.__aclgraph_npu_arch__ = selected_arch
        modules_by_entry[entry_name] = module
    return selected_arch, modules_by_entry


def load_custom_ops_lib(
    *,
    package: str | None = None,
    available_arches: list[str] | None = None,
    get_soc_version=None,
):
    _, modules_by_entry = load_custom_ops_libs(
        package=package,
        available_arches=available_arches,
        get_soc_version=get_soc_version,
    )
    return next(iter(modules_by_entry.values()))
'''
    return (
        template.replace("__MODULE_NAME__", module_name)
        .replace("__ENTRY_MODULES__", repr(entry_modules))
        .replace("__ENTRY_SUPPORTED_ARCHES__", repr(entry_supported_arches))
    )


def render_op_extension_init_py(module_name: str, entries: list[dict[str, Any]]) -> str:
    lines = [
        "from pathlib import Path",
        "",
        "from . import _torch_library",
        "from ._arch_selector import load_custom_ops_libs",
        "",
        "SELECTED_NPU_ARCH, custom_ops_libs = load_custom_ops_libs()",
        "LOADED_LIBRARY_PATHS = {",
    ]
    for entry in entries:
        name = entry["entry_name"]
        lines.append(
            f'    "{name}": Path(custom_ops_libs[{name!r}].__file__).as_posix() if {name!r} in custom_ops_libs else "",'
        )
    lines.extend(
        [
            "}",
            'LOADED_LIBRARY_PATH = next(iter(LOADED_LIBRARY_PATHS.values()), "")',
        ]
    )
    exports = [
        "LOADED_LIBRARY_PATH",
        "LOADED_LIBRARY_PATHS",
        "SELECTED_NPU_ARCH",
        "register_torch_ops",
    ]
    for entry in entries:
        name = entry["entry_name"]
        lines.append(f"def run_{name}(*args, **kwargs):")
        lines.append(f"    if {name!r} not in custom_ops_libs:")
        lines.append(
            f"        raise RuntimeError('ACLGraph custom op {name} is not "
            "packaged for selected NPU arch %s' % SELECTED_NPU_ARCH)"
        )
        lines.append(
            f"    return custom_ops_libs[{name!r}].run_{name}(*args, **kwargs)"
        )
        exports.append(f"run_{name}")
    lines.extend(
        [
            "_torch_library.set_custom_ops_libs(custom_ops_libs)",
            "register_torch_ops = _torch_library.register_torch_ops",
            "register_torch_ops()",
            "",
            f"__all__ = {exports!r}",
            "",
        ]
    )
    return "\n".join(lines)


def render_torch_library_py(module_name: str, entries: list[dict[str, Any]]) -> str:
    entry_supported_arches = {
        str(entry["entry_name"]): _normalize_arch_list(entry.get("supported_arches"))
        for entry in entries
    }
    lines = [
        "import torch",
        "import torch.library as library",
        "",
        "_LIBRARY = None",
        "_REGISTERED = False",
        "_CUSTOM_OPS_LIBS_BY_ENTRY = {}",
        f"_ENTRY_SUPPORTED_ARCHES = {entry_supported_arches!r}",
        "_SELECTED_NPU_ARCH = ''",
        "",
        "",
        "def set_custom_ops_libs(custom_ops_libs_by_entry):",
        "    global _CUSTOM_OPS_LIBS_BY_ENTRY, _SELECTED_NPU_ARCH",
        "    _CUSTOM_OPS_LIBS_BY_ENTRY = dict(custom_ops_libs_by_entry)",
        "    for module in _CUSTOM_OPS_LIBS_BY_ENTRY.values():",
        "        _SELECTED_NPU_ARCH = getattr(module, '__aclgraph_npu_arch__', _SELECTED_NPU_ARCH)",
        "",
        "",
        "def _unsupported_entry_error(entry_name):",
        "    supported = _ENTRY_SUPPORTED_ARCHES.get(entry_name) or []",
        "    supported_text = ', '.join(supported) if supported else '<unspecified>'",
        "    return RuntimeError(",
        "        'ACLGraph custom op %s is not packaged for selected NPU arch %s. Supported arches: %s.'",
        "        % (entry_name, _SELECTED_NPU_ARCH or '<unknown>', supported_text)",
        "    )",
        "",
        "",
        "def set_custom_ops_lib(custom_ops_lib):",
        "    set_custom_ops_libs({",
    ]
    for entry in entries:
        lines.append(f'        "{entry["entry_name"]}": custom_ops_lib,')
    lines.extend(
        [
            "    })",
            "",
            "",
        ]
    )
    lines.extend(
        [
            "def register_torch_ops():",
            "    global _LIBRARY, _REGISTERED",
            "    if _REGISTERED:",
            "        return",
            '    _LIBRARY = library.Library("ascendc_ops", "FRAGMENT")',
        ]
    )
    for entry in entries:
        name = entry["entry_name"]
        schema = _torch_schema(entry)
        run_args = ", ".join(_run_param_names(entry))
        return_arg = _return_tensor_param(entry)
        lines.extend(
            [
                f'    _LIBRARY.define("{schema}")',
                "",
                f'    @library.impl(_LIBRARY, "{name}", "Meta")',
                f"    def {name}_meta({run_args}):",
                f"        return torch.empty_like({return_arg})",
                "",
                f'    @library.impl(_LIBRARY, "{name}", "PrivateUse1")',
                f"    def {name}_impl({run_args}):",
                f"        if {name!r} not in _CUSTOM_OPS_LIBS_BY_ENTRY:",
                f"            raise _unsupported_entry_error({name!r})",
                f"        return _CUSTOM_OPS_LIBS_BY_ENTRY[{name!r}].run_{name}({run_args})",
                "",
            ]
        )
    lines.append("    _REGISTERED = True")
    lines.append("")
    return "\n".join(lines)


def render_setup_py_aclgraph_style(
    module_name: str,
    entries: list[dict[str, Any]],
    *,
    package_name: str = "op_extension",
    package_version: str = "0.1.0",
    operator_build_config: dict[str, Any] | None = None,
    package_data: list[str] | None = None,
) -> str:
    entry_specs = _aclgraph_entry_specs(entries)
    build_config = _setup_build_config(operator_build_config or {})
    template = """import glob
import importlib.util
import os
import sysconfig
from distutils.errors import CompileError
from distutils.spawn import find_executable

import torch
import torch.utils.cpp_extension as cpp_extension
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODULE_BASE = "__MODULE_NAME__"
ENTRY_SPECS = __ENTRY_SPECS__
OPERATOR_BUILD_CONFIG = __OPERATOR_BUILD_CONFIG__
PACKAGE_DATA = __PACKAGE_DATA__

# Source-backed automatic mappings only. The Ascend 950 rule is backed by:
# - Ascend/pytorch torch_npu SoC enum/string handling for Ascend950.
# - cann/asc-devkit examples README mapping Ascend 950 products to dav-3510.
OFFICIAL_AUTO_SOC_ARCH_RULES = (
    {
        "arch": "dav-2201",
        "string_prefixes": ("ascend910b",),
        "enum_values": (104, 220, 221, 222, 223, 224, 225),
    },
    {
        "arch": "dav-3510",
        "string_prefixes": ("ascend950",),
        "enum_values": (260,),
    },
)


def normalize_arch(arch):
    return str(arch or "").strip().lower().replace("_", "-")


def arch_to_module_suffix(arch):
    return normalize_arch(arch).replace("-", "_")


def _split_arches(raw_arches):
    arches = []
    seen = set()
    for item in str(raw_arches or "").replace(";", ",").split(","):
        arch = normalize_arch(item)
        if not arch or arch in seen:
            continue
        seen.add(arch)
        arches.append(arch)
    return arches


def _split_chips(raw_chips):
    chips = []
    seen = set()
    for item in str(raw_chips or "").replace(";", ",").split(","):
        chip = item.strip()
        normalized = chip.lower().replace("_", "").replace("-", "")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        chips.append(chip)
    return chips


def _detect_soc_version():
    try:
        import torch_npu

        npu = getattr(torch_npu, "npu", None)
        getter = getattr(npu, "get_soc_version", None)
        if getter is None:
            return None
        return getter()
    except Exception:
        return None


def _auto_arch_from_soc_version(soc_version):
    if soc_version is None:
        return None
    raw = str(soc_version).strip()
    normalized = raw.lower().replace(" ", "").replace("_", "")
    enum_value = None
    try:
        enum_value = int(raw)
    except (TypeError, ValueError):
        enum_value = None

    for rule in OFFICIAL_AUTO_SOC_ARCH_RULES:
        arch = str(rule["arch"])
        for prefix in rule.get("string_prefixes", ()):
            if normalized.startswith(str(prefix).lower().replace("_", "")):
                return arch
        enum_values = rule.get("enum_values", ())
        if enum_value is not None and enum_value in {int(value) for value in enum_values}:
            return arch
    return None


def _arch_from_chip(chip):
    raw = str(chip or "").strip()
    normalized = raw.lower().replace(" ", "").replace("_", "").replace("-", "")
    enum_value = None
    try:
        enum_value = int(raw)
    except (TypeError, ValueError):
        enum_value = None
    for rule in OFFICIAL_AUTO_SOC_ARCH_RULES:
        arch = str(rule["arch"])
        for prefix in rule.get("string_prefixes", ()):
            if normalized.startswith(str(prefix).lower().replace("_", "").replace("-", "")):
                return arch
        enum_values = rule.get("enum_values", ())
        if enum_value is not None and enum_value in {int(value) for value in enum_values}:
            return arch
    return None


def get_npu_arches():
    raw_target_chips = os.getenv("SK_TARGET_CHIPS")
    if raw_target_chips:
        arches = []
        unsupported = []
        for chip in _split_chips(raw_target_chips):
            arch = _arch_from_chip(chip)
            if arch:
                arches.append(arch)
            else:
                unsupported.append(chip)
        if unsupported:
            raise RuntimeError("Unsupported SK_TARGET_CHIPS values: %s" % ", ".join(unsupported))
        return sorted(dict.fromkeys(arches))

    raw_multi_arches = os.getenv("SK_NPU_ARCHS")
    if raw_multi_arches:
        arches = _split_arches(raw_multi_arches)
        source = "SK_NPU_ARCHS"
    else:
        detected_soc = _detect_soc_version()
        detected_arch = _auto_arch_from_soc_version(detected_soc)
        if detected_arch:
            return [detected_arch]
        raise RuntimeError(
            "No NPU arch specified for ACLGraph custom ops build. Set SK_NPU_ARCHS. "
            "Automatic build detection only uses source-backed SoC mappings; detected soc_version=%r."
            % (detected_soc,)
        )
    if not arches:
        raise RuntimeError("%s is set but contains no valid NPU arch values." % source)
    return arches


def arches_for_entry(entry_spec):
    requested = get_npu_arches()
    declared = _split_arches(",".join(entry_spec.get("supported_arches", [])))
    if not declared:
        return requested
    return [arch for arch in requested if arch in declared]


def get_arch_compile_flags(npu_arch):
    return ["--npu-arch=%s" % npu_arch]


def _config_list(name):
    values = OPERATOR_BUILD_CONFIG.get(name, [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def _config_env(name):
    values = OPERATOR_BUILD_CONFIG.get(name, {})
    if not isinstance(values, dict):
        return {}
    return {str(key): str(value) for key, value in values.items()}


def _compile_definition_flags():
    flags = []
    for value in _config_list("compile_definitions"):
        flags.append(value if value.startswith("-D") else "-D%s" % value)
    return flags


def _link_library_flags():
    flags = []
    for value in _config_list("link_libraries"):
        if value.startswith("-l") or value.startswith("-Wl,") or value.endswith((".so", ".a")):
            flags.append(value)
        else:
            flags.append("-l%s" % value)
    return flags


def _safe_parse_jobs(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        raise RuntimeError("SK_BISHENG_JOBS must be a positive integer")
    if parsed <= 0:
        raise RuntimeError("SK_BISHENG_JOBS must be a positive integer")
    return parsed


def get_dependency_paths():
    python_include = sysconfig.get_config_var("INCLUDEPY")
    python_lib = sysconfig.get_config_var("LIBDIR")
    torch_include_paths = cpp_extension.include_paths()
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    all_include_paths = [BASE_DIR, os.path.join(BASE_DIR, "csrc"), *torch_include_paths, python_include]
    all_include_paths.extend(_config_list("include_dirs"))
    all_libs = [python_lib, torch_lib]
    all_libs.extend(_config_list("link_dirs"))
    torch_npu_spec = importlib.util.find_spec("torch_npu")
    if torch_npu_spec is not None and torch_npu_spec.origin:
        torch_npu_path = os.path.dirname(os.path.realpath(torch_npu_spec.origin))
        all_include_paths.append(os.path.join(torch_npu_path, "include"))
        all_libs.append(os.path.join(torch_npu_path, "lib"))
    return {"all_includes": all_include_paths, "all_libs": all_libs}


class AscendBuildExtension(build_ext):
    def finalize_options(self):
        super().finalize_options()
        requested_jobs = _safe_parse_jobs(os.getenv("SK_BISHENG_JOBS"))
        if requested_jobs is not None:
            self.parallel = min(requested_jobs, max(1, len(self.extensions or [])))

    def _check_bisheng_compiler(self):
        if not find_executable("bisheng"):
            raise RuntimeError("bisheng command not found")

    def build_extension(self, ext):
        self._check_bisheng_compiler()
        for key, value in _config_env("build_env").items():
            os.environ.setdefault(key, value)
        dep_paths = get_dependency_paths()
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)
        abi_value = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
        npu_arch = getattr(ext, "npu_arch", None)
        if not npu_arch:
            raise RuntimeError("internal error: ACLGraph extension is missing npu_arch")
        extension_basename = ext.name.rsplit(".", 1)[-1]
        compile_cmd = [
            "bisheng",
            "-x", "asc",
            *get_arch_compile_flags(npu_arch),
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=%s" % abi_value,
            "-DTORCH_EXTENSION_NAME=%s" % extension_basename,
            "-ltorch_npu", "-ltorch", "-lc10",
        ]
        for force_include in _config_list("force_includes"):
            compile_cmd.extend(["-include", force_include])
        compile_cmd.extend(getattr(ext, "compile_defines", []))
        compile_cmd.extend(_compile_definition_flags())
        compile_cmd.extend(getattr(ext, "compile_options", []))
        compile_cmd.extend(_config_list("compile_options"))
        compile_cmd.extend("-I%s" % include_dir for include_dir in dep_paths["all_includes"] if include_dir)
        compile_cmd.extend("-L%s" % lib_dir for lib_dir in dep_paths["all_libs"] if lib_dir)
        compile_cmd.extend(_link_library_flags())
        compile_cmd.extend(_config_list("link_options"))
        compile_cmd.extend(ext.sources)
        compile_cmd.extend(["-o", ext_fullpath])
        try:
            self.spawn(compile_cmd)
        except Exception as exc:
            raise CompileError(str(exc)) from exc


def make_extension(entry_spec, npu_arch):
    extension_basename = "%s_%s" % (entry_spec["module_base"], arch_to_module_suffix(npu_arch))
    ext = Extension(
        name="__PYTHON_PACKAGE__.%s" % extension_basename,
        sources=[
            os.path.join(BASE_DIR, entry_spec["csrc_file"]),
            os.path.join(BASE_DIR, entry_spec["pybind_file"]),
            *[os.path.join(BASE_DIR, source) for source in entry_spec.get("extra_sources", [])],
        ],
        language="asc",
    )
    ext.npu_arch = npu_arch
    ext.entry_name = entry_spec["entry_name"]
    ext.compile_defines = list(entry_spec.get("compile_defines", []))
    ext.compile_options = list(entry_spec.get("compile_options", []))
    return ext


extensions = [
    make_extension(entry_spec, npu_arch)
    for entry_spec in ENTRY_SPECS
    for npu_arch in arches_for_entry(entry_spec)
]

if not extensions:
    raise RuntimeError("No ACLGraph native extensions selected for requested target chips/arches.")

setup(
    name="__PACKAGE_NAME__",
    version="__PACKAGE_VERSION__",
    ext_modules=extensions,
    packages=find_packages(),
    package_data={"__PYTHON_PACKAGE__": PACKAGE_DATA},
    cmdclass={"build_ext": AscendBuildExtension},
)
"""
    return (
        template.replace("__MODULE_NAME__", module_name)
        .replace("__ENTRY_SPECS__", repr(entry_specs))
        .replace("__OPERATOR_BUILD_CONFIG__", repr(build_config))
        .replace("__PACKAGE_DATA__", repr(package_data or ["_name_resolution.json"]))
        .replace("__PACKAGE_NAME__", package_name)
        .replace("__PYTHON_PACKAGE__", _safe_identifier(package_name, "op_extension"))
        .replace("__PACKAGE_VERSION__", package_version)
    )


def _standalone_param_decls(entry: dict[str, Any]) -> str:
    return ", ".join(
        f"{param.get('c_type', 'GM_ADDR')} {param.get('name', f'arg{index}')}"
        for index, param in enumerate(entry.get("parameters", []))
    )


def _standalone_arg_names(entry: dict[str, Any]) -> str:
    return ", ".join(
        param.get("name", f"arg{index}")
        for index, param in enumerate(entry.get("parameters", []))
    )


def _standalone_default_arg_lines(entry: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for index, param in enumerate(entry.get("parameters", [])):
        c_type = param.get("c_type", "GM_ADDR")
        name = param.get("name", f"arg{index}")
        if "GM_ADDR" in c_type or "*" in c_type or "__gm__" in c_type:
            lines.append(f"    {c_type} {name} = nullptr;")
        else:
            lines.append(f"    {c_type} {name}{{}};")
    return lines


def _standalone_fixture_status(
    entry: dict[str, Any], op_fixture: Path
) -> dict[str, Any]:
    if not op_fixture.exists() or not any(op_fixture.rglob("*")):
        return {
            "status": "insufficient",
            "path": str(op_fixture),
            "device_runnable": False,
        }
    spec_path = op_fixture / "operator-sk-runtime-fixture.json"
    if not spec_path.is_file():
        return {
            "status": "insufficient",
            "path": str(op_fixture),
            "device_runnable": False,
            "reason": "asset fixture copied but no operator-sk-runtime-fixture.json device plan",
        }
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "insufficient",
            "path": str(op_fixture),
            "device_runnable": False,
            "reason": f"operator-sk-runtime-fixture.json is invalid JSON: {exc}",
        }
    params = spec.get("parameters")
    if not isinstance(params, list):
        return {
            "status": "insufficient",
            "path": str(op_fixture),
            "device_runnable": False,
            "reason": "operator-sk-runtime-fixture.json must contain a parameters list",
        }
    by_name = {item.get("name"): item for item in params if isinstance(item, dict)}
    missing = [
        param.get("name")
        for param in entry.get("parameters", [])
        if param.get("name") not in by_name
    ]
    if missing:
        return {
            "status": "insufficient",
            "path": str(op_fixture),
            "device_runnable": False,
            "reason": "operator-sk-runtime-fixture.json missing parameters: "
            + ", ".join(str(item) for item in missing),
        }
    has_compare_buffer = False
    for param in entry.get("parameters", []):
        name = param.get("name")
        spec_item = by_name.get(name)
        if not isinstance(spec_item, dict):
            continue
        kind = spec_item.get("kind", "device_buffer")
        if kind == "device_buffer":
            bytes_value = spec_item.get("bytes")
            if (
                not isinstance(bytes_value, int)
                or isinstance(bytes_value, bool)
                or bytes_value <= 0
            ):
                return {
                    "status": "insufficient",
                    "path": str(op_fixture),
                    "device_runnable": False,
                    "reason": f"operator-sk-runtime-fixture.json has invalid byte size for {name}",
                }
            has_compare_buffer = has_compare_buffer or bool(
                spec_item.get("compare", False)
            )
        elif kind not in {"scalar", "literal"}:
            return {
                "status": "insufficient",
                "path": str(op_fixture),
                "device_runnable": False,
                "reason": f"operator-sk-runtime-fixture.json has unsupported kind for {name}: {kind}",
            }
    if not has_compare_buffer:
        return {
            "status": "insufficient",
            "path": str(op_fixture),
            "device_runnable": False,
            "reason": "operator-sk-runtime-fixture.json must mark at least one device buffer with compare=true",
        }
    return {
        "status": "available",
        "path": str(op_fixture),
        "device_runnable": True,
        "spec": spec,
    }


def _cpp_json_string(value: str) -> str:
    return json.dumps(value)


def _cpp_scalar_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"unsupported scalar fixture literal: {value!r}")


def _standalone_buffer_cast(c_type: str, variable: str) -> str:
    if "GM_ADDR" in c_type:
        return variable
    if "*" in c_type or "__gm__" in c_type:
        return f"reinterpret_cast<{c_type}>({variable})"
    return variable


def _fixture_param_by_name(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["name"]: item
        for item in fixture.get("parameters", [])
        if isinstance(item, dict) and "name" in item
    }


def _render_real_device_compare_body(
    entry: dict[str, Any], fixture: dict[str, Any]
) -> list[str]:
    wrapper = _cpp_identifier_from_symbol(entry["entry_name"])
    params_by_name = _fixture_param_by_name(fixture)
    block_dim = int(fixture.get("block_dim", 1))
    lines = [
        '    result.reason = "standalone_bind_target_baseline_and_sk_outputs_matched";',
        f"    uint32_t entry_block_dim = {block_dim};",
        "    std::vector<void *> cleanup_device_ptrs;",
    ]
    compare_buffers: list[tuple[str, str]] = []
    baseline_args: list[str] = []
    sk_args: list[str] = []
    for index, param in enumerate(entry.get("parameters", [])):
        name = param.get("name", f"arg{index}")
        c_type = param.get("c_type", "GM_ADDR")
        spec = params_by_name[name]
        kind = spec.get("kind", "device_buffer")
        if kind == "device_buffer":
            bytes_value = int(spec["bytes"])
            fill_value = int(spec.get("fill", spec.get("input_fill", 0))) & 0xFF
            baseline_var = f"{name}_baseline_device"
            sk_var = f"{name}_sk_device"
            host_var = f"{name}_host"
            lines.extend(
                [
                    f"    const size_t {name}_bytes = {bytes_value};",
                    f"    std::vector<uint8_t> {host_var}({name}_bytes, static_cast<uint8_t>({fill_value}));",
                    f"    uint8_t *{baseline_var} = nullptr;",
                    f"    uint8_t *{sk_var} = nullptr;",
                    (
                        f"    if (!acl_ok(aclrtMalloc(reinterpret_cast<void **>"
                        f"(&{baseline_var}), {name}_bytes, "
                        f'ACL_MEM_MALLOC_HUGE_FIRST), result, "aclrtMalloc {name} '
                        'baseline")) { free_device_ptrs(cleanup_device_ptrs); '
                        "return result; }"
                    ),
                    f"    cleanup_device_ptrs.push_back(static_cast<void *>({baseline_var}));",
                    (
                        f"    if (!acl_ok(aclrtMalloc(reinterpret_cast<void **>"
                        f"(&{sk_var}), {name}_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "
                        f'result, "aclrtMalloc {name} sk")) {{ '
                        "free_device_ptrs(cleanup_device_ptrs); return result; }"
                    ),
                    f"    cleanup_device_ptrs.push_back(static_cast<void *>({sk_var}));",
                    (
                        f"    if (!acl_ok(aclrtMemcpy({baseline_var}, {name}_bytes, "
                        f"{host_var}.data(), {name}_bytes, "
                        f'ACL_MEMCPY_HOST_TO_DEVICE), result, "aclrtMemcpy {name} '
                        'baseline input")) { free_device_ptrs(cleanup_device_ptrs); '
                        "return result; }"
                    ),
                    (
                        f"    if (!acl_ok(aclrtMemcpy({sk_var}, {name}_bytes, "
                        f"{host_var}.data(), {name}_bytes, "
                        f'ACL_MEMCPY_HOST_TO_DEVICE), result, "aclrtMemcpy {name} '
                        'sk input")) { free_device_ptrs(cleanup_device_ptrs); '
                        "return result; }"
                    ),
                ]
            )
            baseline_args.append(_standalone_buffer_cast(c_type, baseline_var))
            sk_args.append(_standalone_buffer_cast(c_type, sk_var))
            if spec.get("compare", False):
                compare_buffers.append((name, bytes_value))
        elif kind in {"scalar", "literal"}:
            literal = _cpp_scalar_literal(spec.get("value", 0))
            lines.append(f"    {c_type} {name} = {literal};")
            baseline_args.append(name)
            sk_args.append(name)
        else:
            raise ValueError(
                f"unsupported runtime fixture kind for {entry['entry_name']}.{name}: {kind}"
            )
    baseline_arg_suffix = f"{', ' if baseline_args else ''}{', '.join(baseline_args)}"
    lines.append(
        f"    launch_{wrapper}_baseline(entry_block_dim, stream{baseline_arg_suffix});"
    )
    lines.append(
        "    if (!acl_ok(aclrtSynchronizeStream(acl_stream), result, "
        '"aclrtSynchronizeStream baseline")) { '
        "free_device_ptrs(cleanup_device_ptrs); return result; }"
    )
    sk_arg_suffix = f"{', ' if sk_args else ''}{', '.join(sk_args)}"
    lines.append(
        f"    if (!launch_{wrapper}_sk(entry_block_dim, stream, "
        f"result{sk_arg_suffix})) {{ free_device_ptrs(cleanup_device_ptrs); "
        "return result; }"
    )
    lines.append(
        "    if (!acl_ok(aclrtSynchronizeStream(acl_stream), result, "
        '"aclrtSynchronizeStream sk")) { free_device_ptrs(cleanup_device_ptrs); '
        "return result; }"
    )
    if compare_buffers:
        lines.append("    bool matched = true;")
        for name, _bytes_value in compare_buffers:
            lines.extend(
                [
                    f"    std::vector<uint8_t> {name}_baseline_host({name}_bytes);",
                    f"    std::vector<uint8_t> {name}_sk_host({name}_bytes);",
                    (
                        f"    if (!acl_ok(aclrtMemcpy({name}_baseline_host.data(), "
                        f"{name}_bytes, {name}_baseline_device, {name}_bytes, "
                        f'ACL_MEMCPY_DEVICE_TO_HOST), result, "aclrtMemcpy {name} '
                        'baseline output")) { '
                        "free_device_ptrs(cleanup_device_ptrs); return result; }"
                    ),
                    (
                        f"    if (!acl_ok(aclrtMemcpy({name}_sk_host.data(), "
                        f"{name}_bytes, {name}_sk_device, {name}_bytes, "
                        f'ACL_MEMCPY_DEVICE_TO_HOST), result, "aclrtMemcpy {name} '
                        'sk output")) { free_device_ptrs(cleanup_device_ptrs); '
                        "return result; }"
                    ),
                    f"    result.baseline_hash = hash_combine(result.baseline_hash, fnv1a64({name}_baseline_host));",
                    f"    result.sk_hash = hash_combine(result.sk_hash, fnv1a64({name}_sk_host));",
                    f"    result.compared_bytes += {name}_bytes;",
                    f"    matched = matched && ({name}_baseline_host == {name}_sk_host);",
                ]
            )
        lines.extend(
            [
                "    if (!matched) {",
                '        result.status = "failed";',
                '        result.reason = "differential_outputs_mismatched";',
                "    }",
            ]
        )
    else:
        lines.append('    result.status = "skipped-insufficient-runtime-spec";')
        lines.append(
            '    result.reason = "standalone_bind_target_baseline_and_sk_executed_no_compare_buffer";'
        )
    lines.append("    free_device_ptrs(cleanup_device_ptrs);")
    return lines


def render_standalone_compare_source(
    adapted_manifest: dict[str, Any],
    runtime_fixtures: dict[str, Any] | None = None,
    *,
    source_stem: str | None = None,
    include_kernel: bool = True,
) -> str:
    """Render a standalone ASC executable source for bind-target differential routing."""
    entries = _manifest_entries(adapted_manifest)
    if source_stem is not None:
        filtered_entries = []
        for entry in entries:
            if entry["entry_name"] == source_stem:
                filtered_entries.append(entry)
        entries = filtered_entries
    csrc_includes = []
    for item in adapted_manifest.get("canonical_written_files", []):
        item_text = str(item)
        item_path = Path(item_text)
        if not item_text.startswith("operator-sk-adapted/csrc/"):
            continue
        if not item_text.endswith(".asc"):
            continue
        if _is_aclgraph_pybind_source_name(item_path.name):
            continue
        if source_stem is not None and item_path.stem != source_stem:
            continue
        csrc_includes.append(f"csrc/{item_path.stem}/{item_path.name}")
    csrc_includes = sorted(csrc_includes)
    if not include_kernel:
        csrc_includes = []
    lines = [
        "#include <cstdint>",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <iostream>",
        "#include <string>",
        "#include <vector>",
        '#include "acl/acl.h"',
        '#include "kernel_operator.h"',
        '#include "super_kernel/super_kernel.h"',
        "",
    ]
    for csrc in csrc_includes:
        lines.append(f'#include "{csrc}"')
    if csrc_includes:
        lines.append("")
    lines.extend(
        [
            "struct StandaloneEntryResult {",
            "    std::string entry_name;",
            "    std::string status;",
            "    std::string reason;",
            "    std::string baseline_call;",
            "    std::string sk_call;",
            "    uint64_t baseline_hash = 0;",
            "    uint64_t sk_hash = 0;",
            "    size_t compared_bytes = 0;",
            "};",
            "",
            "static bool standalone_env_enabled(const char *name)",
            "{",
            "    const char *value = std::getenv(name);",
            "    return value != nullptr && value[0] != '\\0' && std::string(value) != \"0\";",
            "}",
            "",
            "static void print_json_string(const std::string &value)",
            "{",
            '    std::cout << "\\"";',
            "    for (char ch : value) {",
            "        if (ch == '\\\\' || ch == '\\\"') {",
            "            std::cout << '\\\\';",
            "        }",
            "        std::cout << ch;",
            "    }",
            '    std::cout << "\\"";',
            "}",
            "",
            "static uint64_t fnv1a64(const std::vector<uint8_t> &value)",
            "{",
            "    uint64_t hash = 1469598103934665603ULL;",
            "    for (uint8_t byte : value) {",
            "        hash ^= static_cast<uint64_t>(byte);",
            "        hash *= 1099511628211ULL;",
            "    }",
            "    return hash;",
            "}",
            "",
            "static uint64_t hash_combine(uint64_t left, uint64_t right)",
            "{",
            "    return left ^ (right + 0x9e3779b97f4a7c15ULL + (left << 6) + (left >> 2));",
            "}",
            "",
            "static bool acl_ok(aclError code, StandaloneEntryResult &result, const std::string &step)",
            "{",
            "    if (code == ACL_ERROR_NONE) {",
            "        return true;",
            "    }",
            '    result.status = "failed";',
            '    result.reason = step + ":acl_error_" + std::to_string(static_cast<int>(code));',
            "    return false;",
            "}",
            "",
            "static void free_device_ptrs(const std::vector<void *> &ptrs)",
            "{",
            "    for (void *ptr : ptrs) {",
            "        if (ptr != nullptr) {",
            "            aclrtFree(ptr);",
            "        }",
            "    }",
            "}",
            "",
            "static std::string overall_status(const std::vector<StandaloneEntryResult> &results)",
            "{",
            "    bool has_failed = false;",
            "    bool has_passed = false;",
            "    bool has_skipped_no_npu = false;",
            "    bool has_skipped_insufficient = false;",
            "    for (const auto &result : results) {",
            '        has_failed = has_failed || result.status == "failed";',
            '        has_passed = has_passed || result.status == "passed";',
            '        has_skipped_no_npu = has_skipped_no_npu || result.status == "skipped-no-npu";',
            "        has_skipped_insufficient = has_skipped_insufficient || "
            'result.status == "skipped-insufficient-runtime-spec";',
            "    }",
            '    if (has_failed) return "failed";',
            '    if (has_passed && !has_skipped_no_npu && !has_skipped_insufficient) return "passed";',
            '    if (!has_passed && has_skipped_no_npu && !has_skipped_insufficient) return "skipped-no-npu";',
            "    if (!has_passed && !has_skipped_no_npu && "
            'has_skipped_insufficient) return "skipped-insufficient-runtime-spec";',
            '    return "mixed";',
            "}",
            "",
            "static void print_results(const std::vector<StandaloneEntryResult> &results)",
            "{",
            '    std::cout << "{\\"backend\\":\\"standalone\\",\\"status\\":";',
            "    print_json_string(overall_status(results));",
            '    std::cout << ",\\"outputs\\":{";',
            "    for (size_t i = 0; i < results.size(); ++i) {",
            "        if (i != 0) {",
            '            std::cout << ",";',
            "        }",
            "        print_json_string(results[i].entry_name);",
            '        std::cout << ":{\\"baseline\\":[" << results[i].baseline_hash '
            '<< "," << results[i].compared_bytes << "],\\"sk\\":[" '
            '<< results[i].sk_hash << "," << results[i].compared_bytes << "]}";',
            "    }",
            '    std::cout << "},\\"calls\\":{";',
            "    for (size_t i = 0; i < results.size(); ++i) {",
            "        if (i != 0) {",
            '            std::cout << ",";',
            "        }",
            "        print_json_string(results[i].entry_name);",
            '        std::cout << ":[";',
            "        print_json_string(results[i].baseline_call);",
            '        std::cout << ",";',
            "        print_json_string(results[i].sk_call);",
            '        std::cout << "]";',
            "    }",
            '    std::cout << "},\\"statuses\\":{";',
            "    for (size_t i = 0; i < results.size(); ++i) {",
            "        if (i != 0) {",
            '            std::cout << ",";',
            "        }",
            "        print_json_string(results[i].entry_name);",
            '        std::cout << ":{\\"status\\":";',
            "        print_json_string(results[i].status);",
            '        std::cout << ",\\"reason\\":";',
            "        print_json_string(results[i].reason);",
            '        std::cout << "}";',
            "    }",
            '    std::cout << "}}" << std::endl;',
            "}",
            "",
        ]
    )
    for entry in entries:
        params = _standalone_param_decls(entry)
        args = _standalone_arg_names(entry)
        bind_target = entry.get("bind_target") or entry["entry_name"]
        wrapper = _cpp_identifier_from_symbol(entry["entry_name"])
        param_tail = f", {params}" if params else ""
        arg_tail = f"{args}" if args else ""
        if include_kernel:
            lines.append(
                f"void launch_{wrapper}_baseline(uint32_t block_dim, void *stream{param_tail})"
            )
            lines.append("{")
            lines.append(
                f"    {bind_target}<<<block_dim, nullptr, stream>>>({arg_tail});"
            )
            lines.append("}")
            lines.append("")
            lines.append(
                f"bool launch_{wrapper}_sk(uint32_t block_dim, void *stream, StandaloneEntryResult &result{param_tail})"
            )
            lines.append("{")
            lines.append(
                "    aclrtStream acl_stream = static_cast<aclrtStream>(stream);"
            )
            lines.append("    aclmdlRI model_ri;")
            lines.append(
                "    if (!acl_ok(aclmdlRICaptureBegin(acl_stream, "
                "ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), result, "
                '"aclmdlRICaptureBegin sk")) return false;'
            )
            lines.append(
                f"    {bind_target}<<<block_dim, nullptr, stream>>>({arg_tail});"
            )
            lines.append(
                "    if (!acl_ok(aclmdlRICaptureEnd(acl_stream, &model_ri), "
                'result, "aclmdlRICaptureEnd sk")) return false;'
            )
            lines.append(
                '    if (!acl_ok(aclskOptimize(model_ri, nullptr), result, "aclskOptimize sk")) return false;'
            )
            lines.append(
                "    if (!acl_ok(aclmdlRIExecuteAsync(model_ri, acl_stream), "
                'result, "aclmdlRIExecuteAsync sk")) return false;'
            )
            lines.append("    return true;")
            lines.append("}")
            lines.append("")
            lines.append(
                f"static void device_launch_{wrapper}(uint32_t block_dim, void *stream)"
            )
            lines.append("{")
            lines.extend(_standalone_default_arg_lines(entry))
            call_args = (", " + args) if args else ""
            lines.append(
                '    StandaloneEntryResult result{"device_launch", "passed", "device_launch_route", "baseline", "sk"};'
            )
            lines.append(
                f"    launch_{wrapper}_baseline(block_dim, stream{call_args});"
            )
            lines.append(
                f"    launch_{wrapper}_sk(block_dim, stream, result{call_args});"
            )
            lines.append("}")
            lines.append("")
        lines.append(
            f"static StandaloneEntryResult compare_{wrapper}(uint32_t block_dim, void *stream)"
        )
        lines.append("{")
        lines.append("    aclrtStream acl_stream = static_cast<aclrtStream>(stream);")
        lines.append("    StandaloneEntryResult result{")
        lines.append(f"        {_cpp_json_string(entry['entry_name'])},")
        lines.append('        "passed",')
        lines.append('        "standalone_mock_bind_target_route_matched",')
        lines.append(f'        "launch_{wrapper}_baseline",')
        lines.append(f'        "launch_{wrapper}_sk"')
        lines.append("    };")
        fixture = (runtime_fixtures or {}).get(entry["entry_name"], {})
        if fixture.get("device_runnable") and isinstance(fixture.get("spec"), dict):
            lines.append(
                '    if (standalone_env_enabled("SK_OPERATOR_RUN_DEVICE_COMPARE")) {'
            )
            lines.extend(_render_real_device_compare_body(entry, fixture["spec"]))
            lines.append("    }")
        else:
            reason = fixture.get(
                "reason",
                "runtime fixture does not include a device-runnable parameter plan",
            )
            lines.append('    result.status = "skipped-insufficient-runtime-spec";')
            lines.append(f"    result.reason = {_cpp_json_string(str(reason))};")
        lines.append("    return result;")
        lines.append("}")
        lines.append("")
    lines.extend(
        [
            "int main()",
            "{",
            '    if (!standalone_env_enabled("SK_OPERATOR_MOCK_NPU") && '
            '!standalone_env_enabled("SK_OPERATOR_RUN_DEVICE_COMPARE")) {',
            '        std::cout << "{\\"backend\\":\\"standalone\\",'
            '\\"status\\":\\"skipped-no-npu\\",\\"outputs\\":{},'
            '\\"calls\\":{},\\"statuses\\":{}}" << std::endl;',
            "        return 0;",
            "    }",
            "    int32_t device_id = 0;",
            '    if (const char *npu_device_id = std::getenv("NPU_DEVICE_ID")) {',
            "        device_id = std::atoi(npu_device_id);",
            '    } else if (const char *ascend_device_id = std::getenv("ASCEND_DEVICE_ID")) {',
            "        device_id = std::atoi(ascend_device_id);",
            "    }",
            "    std::vector<StandaloneEntryResult> results;",
            "    aclrtStream acl_stream = nullptr;",
            '    if (standalone_env_enabled("SK_OPERATOR_RUN_DEVICE_COMPARE")) {',
            "        StandaloneEntryResult runtime_setup_result"
            '{"runtime_setup", "failed", "runtime_setup_failed", '
            '"acl_runtime_setup", "acl_runtime_setup"};',
            '        if (!acl_ok(aclInit(nullptr), runtime_setup_result, "aclInit")) '
            "{ results.push_back(runtime_setup_result); print_results(results); "
            "return 1; }",
            "        if (!acl_ok(aclrtSetDevice(device_id), runtime_setup_result, "
            '"aclrtSetDevice")) { results.push_back(runtime_setup_result); '
            "print_results(results); aclFinalize(); return 1; }",
            "        if (!acl_ok(aclrtCreateStream(&acl_stream), runtime_setup_result, "
            '"aclrtCreateStream")) { results.push_back(runtime_setup_result); '
            "print_results(results); aclrtResetDevice(device_id); aclFinalize(); "
            "return 1; }",
            "    }",
            "    uint32_t block_dim = 1;",
            "    void *stream = static_cast<void *>(acl_stream);",
        ]
    )
    for entry in entries:
        wrapper = _cpp_identifier_from_symbol(entry["entry_name"])
        lines.append(f"    results.push_back(compare_{wrapper}(block_dim, stream));")
    lines.extend(
        [
            "    print_results(results);",
            '    if (standalone_env_enabled("SK_OPERATOR_RUN_DEVICE_COMPARE")) {',
            "        aclrtDestroyStream(acl_stream);",
            "        aclrtResetDevice(device_id);",
            "        aclFinalize();",
            "    }",
            '    if (overall_status(results) == "failed") {',
            "        return 1;",
            "    }",
            "    return 0;",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_standalone_cmake(
    *,
    npu_arch: str,
    runtime_sources: dict[str, str] | None = None,
    operator_build_config: dict[str, Any] | None = None,
) -> str:
    resolved_npu_arch = str(npu_arch or "").strip()
    if not resolved_npu_arch:
        raise ValueError("standalone CMake rendering requires an explicit --npu-arch")
    targets = runtime_sources or {"runtime_compare": "runtime_compare.asc"}
    add_executables = "\n\n".join(
        f"""add_executable({target}
    {source}
)"""
        for target, source in targets.items()
    )
    target_items = " ".join(targets)
    build_config = operator_build_config or {}
    extra_include_dirs = _string_list_from_config(build_config, "include_dirs")
    extra_link_dirs = _string_list_from_config(build_config, "link_dirs")
    extra_link_libraries = _string_list_from_config(build_config, "link_libraries")
    extra_link_options = _string_list_from_config(build_config, "link_options")
    extra_compile_options = _string_list_from_config(build_config, "compile_options")
    extra_force_includes = _string_list_from_config(build_config, "force_includes")
    compile_definitions = _string_list_from_config(build_config, "compile_definitions")
    include_dir_lines = "\n".join(
        f"        {_cmake_quoted(path)}" for path in extra_include_dirs
    )
    link_dir_block = ""
    if extra_link_dirs:
        link_dir_lines = "\n".join(
            f"        {_cmake_quoted(path)}" for path in extra_link_dirs
        )
        link_dir_block = f"""
    target_link_directories(${{runtime_target}} PRIVATE
{link_dir_lines}
    )
"""
    link_library_lines = "\n".join(
        f"        {_cmake_quoted(value)}" for value in extra_link_libraries
    )
    link_option_lines = "\n".join(
        f"        {_cmake_quoted(value)}" for value in extra_link_options
    )
    compile_option_items = []
    for value in compile_definitions:
        compile_option_items.append(value if value.startswith("-D") else f"-D{value}")
    compile_option_items.extend(extra_compile_options)
    for force_include in extra_force_includes:
        compile_option_items.extend(["-include", force_include])
    compile_option_lines = "\n".join(
        f"        {_cmake_quoted(value)}" for value in compile_option_items
    )
    if include_dir_lines:
        include_dir_lines = "\n" + include_dir_lines
    if link_library_lines:
        link_library_lines = "\n" + link_library_lines
    if link_option_lines:
        link_option_lines = "\n" + link_option_lines
    if compile_option_lines:
        compile_option_lines = "\n" + compile_option_lines
    return f"""cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(sk_standalone_compare LANGUAGES ASC CXX)

{add_executables}

file(GLOB_RECURSE STANDALONE_INCLUDE_CANDIDATES LIST_DIRECTORIES true
    "${{CMAKE_CURRENT_SOURCE_DIR}}/csrc/*"
)
set(STANDALONE_INCLUDE_DIRS "${{CMAKE_CURRENT_SOURCE_DIR}}/csrc")
foreach(candidate ${{STANDALONE_INCLUDE_CANDIDATES}})
    if(IS_DIRECTORY "${{candidate}}")
        list(APPEND STANDALONE_INCLUDE_DIRS "${{candidate}}")
    endif()
endforeach()

foreach(runtime_target IN ITEMS {target_items})
    target_include_directories(${{runtime_target}} PRIVATE
        ${{STANDALONE_INCLUDE_DIRS}}
{include_dir_lines}
    )
{link_dir_block}

    target_link_libraries(${{runtime_target}} PRIVATE
        ascendsk acl_rt acl_rtc ascendcl ascend_dump runtime error_manager
        c_sec unified_dlog ascendc_runtime profapi profimpl msprofiler
        stdc++ pthread
{link_library_lines}
    )

    target_link_options(${{runtime_target}} PRIVATE
        -Wl,--allow-shlib-undefined
{link_option_lines}
    )

    target_compile_options(${{runtime_target}} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch={resolved_npu_arch}>
{compile_option_lines}
    )
endforeach()
"""


def _standalone_target_arch_resolution(
    *, target_chip: str, npu_arch: str
) -> dict[str, Any]:
    explicit_arch = str(npu_arch or "").strip()
    if explicit_arch:
        return {
            "status": "resolved",
            "npu_arch": explicit_arch,
            "source": "explicit-npu-arch",
            "target_chip": target_chip,
            "resolved": [],
            "unsupported": [],
        }
    if not str(target_chip or "").strip():
        return {
            "status": "needs-target-arch",
            "npu_arch": "",
            "reason": "missing-npu-arch-and-target-chip",
            "message": "Pass --npu-arch explicitly or pass a single source-backed --target-chip.",
            "target_chip": target_chip,
            "resolved": [],
            "unsupported": [],
        }
    from operator_target_arch import resolve_target_chips

    resolved, unsupported = resolve_target_chips(target_chip)
    arches = sorted(
        {
            str(item.get("arch", "")).strip()
            for item in resolved
            if str(item.get("arch", "")).strip()
        }
    )
    if unsupported:
        return {
            "status": "needs-target-arch",
            "npu_arch": "",
            "reason": "unsupported-target-chip",
            "message": "Target chip cannot be resolved to a source-backed NPU arch; pass --npu-arch explicitly.",
            "target_chip": target_chip,
            "resolved": resolved,
            "unsupported": unsupported,
        }
    if len(arches) != 1:
        return {
            "status": "needs-target-arch",
            "npu_arch": "",
            "reason": "ambiguous-target-chip-arches"
            if arches
            else "missing-target-chip-arch",
            "message": "Target chip must resolve to exactly one source-backed NPU arch; pass --npu-arch explicitly.",
            "target_chip": target_chip,
            "resolved": resolved,
            "unsupported": unsupported,
        }
    return {
        "status": "resolved",
        "npu_arch": arches[0],
        "source": "target-chip-source-backed",
        "target_chip": target_chip,
        "resolved": resolved,
        "unsupported": unsupported,
    }


def _strip_aclgraph_pybind_wrappers(source_text: str) -> str:
    stripped = source_text
    namespace_marker = "\nnamespace ascendc_ops {"
    if namespace_marker in stripped:
        stripped = stripped.split(namespace_marker, 1)[0].rstrip() + "\n"
    removable_includes = [
        "#include <pybind11/pybind11.h>",
        "#include <torch/extension.h>",
        "#include <limits>",
        '#include "third_party/acl/inc/acl/acl_rt.h"',
        '#include "torch_npu/csrc/core/npu/NPUStream.h"',
    ]
    lines = []
    for line in stripped.splitlines():
        if line.strip() not in removable_includes:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _copy_standalone_support_tree(source_dir: Path, dest_dir: Path) -> list[str]:
    copied: list[str] = []
    if not source_dir.is_dir():
        return copied
    for source_path in sorted(item for item in source_dir.rglob("*") if item.is_file()):
        rel = source_path.relative_to(source_dir)
        dest_path = dest_dir / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        copied.append(rel.as_posix())
    return copied


def generate_standalone_compare_artifacts(
    aggregate_output_dir: Path,
    output_root: Path,
    *,
    runtime_fixture_dir: Path | None = None,
    target_chip: str = "",
    npu_arch: str = "",
    operator_build_config: Path | None = None,
) -> dict[str, Any]:
    manifest_path = aggregate_output_dir / "operator-sk-adapted.json"
    source_dir = aggregate_output_dir / "operator-sk-adapted"
    if not manifest_path.is_file():
        raise ValueError(
            f"operator-sk-adapted.json not found in {aggregate_output_dir}"
        )
    if not source_dir.is_dir():
        raise ValueError(
            f"operator-sk-adapted directory not found in {aggregate_output_dir}"
        )
    adapted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if adapted_manifest.get("pybind_layout") != "aclgraph-canonical":
        raise ValueError(
            "standalone compare requires aclgraph-canonical adapted output"
        )
    build_config = _load_operator_build_config(operator_build_config)
    target_arch_resolution = _standalone_target_arch_resolution(
        target_chip=target_chip, npu_arch=npu_arch
    )
    resolved_npu_arch = str(target_arch_resolution.get("npu_arch", "") or "")

    verify_dir = output_root / "operator-sk-standalone-verify"
    if verify_dir.exists():
        raise ValueError(f"standalone verify output already exists: {verify_dir}")
    csrc_dest = verify_dir / "csrc"
    csrc_dest.mkdir(parents=True)
    copied_sources: list[str] = []
    copied_support: dict[str, list[str]] = {}
    source_paths = sorted(
        source_path
        for source_path in (source_dir / "csrc").glob("*.asc")
        if not _is_aclgraph_pybind_source_name(source_path.name)
    )
    entries = _manifest_entries(adapted_manifest)
    fixture_statuses: dict[str, Any] = {}
    if runtime_fixture_dir is not None and runtime_fixture_dir.exists():
        for entry in entries:
            op_fixture = runtime_fixture_dir / entry["entry_name"]
            fixture_statuses[entry["entry_name"]] = _standalone_fixture_status(
                entry, op_fixture
            )

    runtime_sources: dict[str, str] = {}
    entry_targets: dict[str, str] = {}
    single_source = len(source_paths) == 1
    for source_path in source_paths:
        if _is_aclgraph_pybind_source_name(source_path.name):
            continue
        source_dest_dir = csrc_dest / source_path.stem
        source_dest_dir.mkdir(parents=True, exist_ok=True)
        kernel_only = _strip_aclgraph_pybind_wrappers(
            source_path.read_text(encoding="utf-8")
        )
        (source_dest_dir / source_path.name).write_text(kernel_only, encoding="utf-8")
        copied_sources.append(
            f"operator-sk-standalone-verify/csrc/{source_path.stem}/{source_path.name}"
        )
        if runtime_fixture_dir is not None:
            support_source = runtime_fixture_dir / source_path.stem / "asset-support"
            support_files = _copy_standalone_support_tree(
                support_source, source_dest_dir
            )
            if support_files:
                copied_support[source_path.stem] = [
                    f"operator-sk-standalone-verify/csrc/{source_path.stem}/{rel}"
                    for rel in support_files
                ]
        source_entries = []
        for entry in entries:
            if entry["entry_name"] == source_path.stem:
                source_entries.append(entry)
        has_device_runnable = any(
            fixture_statuses.get(entry["entry_name"], {}).get("device_runnable") is True
            for entry in source_entries
        )
        include_kernel = runtime_fixture_dir is None or has_device_runnable
        runtime_source_name = (
            "runtime_compare.asc"
            if single_source
            else f"runtime_compare_{source_path.stem}.asc"
        )
        target_name = (
            "runtime_compare"
            if single_source
            else f"runtime_compare_{_cpp_identifier_from_symbol(source_path.stem)}"
        )
        (verify_dir / runtime_source_name).write_text(
            render_standalone_compare_source(
                adapted_manifest,
                fixture_statuses,
                source_stem=source_path.stem,
                include_kernel=include_kernel,
            ),
            encoding="utf-8",
        )
        runtime_sources[target_name] = (
            f"operator-sk-standalone-verify/{runtime_source_name}"
        )
        for entry in source_entries:
            entry_targets[entry["entry_name"]] = target_name
    if target_arch_resolution["status"] == "resolved":
        (verify_dir / "CMakeLists.txt").write_text(
            render_standalone_cmake(
                npu_arch=resolved_npu_arch,
                runtime_sources={k: Path(v).name for k, v in runtime_sources.items()},
                operator_build_config=build_config,
            ),
            encoding="utf-8",
        )
    primary_runtime = next(
        iter(runtime_sources.values()),
        "operator-sk-standalone-verify/runtime_compare.asc",
    )
    manifest = {
        "status": "generated"
        if target_arch_resolution["status"] == "resolved"
        else "needs-target-arch",
        "aggregate_output_dir": str(aggregate_output_dir),
        "standalone_verify_dir": str(verify_dir),
        "target_chip": target_chip,
        "npu_arch": resolved_npu_arch,
        "target_arch_resolution": target_arch_resolution,
        "operator_build_config": build_config,
        "runtime_compare": primary_runtime,
        "runtime_sources": runtime_sources,
        "entry_targets": entry_targets,
        "cmake": "operator-sk-standalone-verify/CMakeLists.txt"
        if target_arch_resolution["status"] == "resolved"
        else "",
        "copied_sources": copied_sources,
        "copied_support": copied_support,
        "entries": entries,
        "runtime_fixtures": fixture_statuses,
    }
    if target_arch_resolution["status"] != "resolved":
        manifest["reason"] = target_arch_resolution.get("reason", "needs-target-arch")
        manifest["message"] = target_arch_resolution.get("message", "")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "operator-sk-standalone-verify.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _safe_identifier(text: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", text).strip("_")
    if not safe:
        safe = fallback
    if safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def _cmake_quoted(value: str) -> str:
    return '"' + str(value).replace("\\", "/").replace('"', '\\"') + '"'


def _load_operator_build_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {
            "schema_version": "sk.operator.build_config.resolved.v1",
            "status": "ready",
        }
    if not config_path.is_file():
        raise ValueError(f"resolved operator build config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("resolved operator build config must be a JSON object")
    return payload


def _string_list_from_config(config: dict[str, Any], config_key: str) -> list[str]:
    raw_value = config.get(config_key, [])
    if not isinstance(raw_value, list):
        return []
    return [str(item) for item in raw_value if str(item)]


def _env_from_config(config: dict[str, Any], config_key: str) -> dict[str, str]:
    raw_value = config.get(config_key, {})
    if not isinstance(raw_value, dict):
        return {}
    return {str(key): str(value) for key, value in raw_value.items()}


def _setup_build_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "include_dirs": _string_list_from_config(config, "include_dirs"),
        "force_includes": _string_list_from_config(config, "force_includes"),
        "compile_options": _string_list_from_config(config, "compile_options"),
        "compile_definitions": _string_list_from_config(config, "compile_definitions"),
        "link_dirs": _string_list_from_config(config, "link_dirs"),
        "link_libraries": _string_list_from_config(config, "link_libraries"),
        "link_options": _string_list_from_config(config, "link_options"),
        "build_env": _env_from_config(config, "build_env"),
        "runtime_env": _env_from_config(config, "runtime_env"),
    }


def _copy_operator_package_files(
    package_dir: Path, build_config: dict[str, Any]
) -> list[str]:
    copied: list[str] = []
    resources_dir = package_dir / "_resources"
    used_names: set[str] = set()
    for raw_path in _string_list_from_config(build_config, "package_files"):
        source = Path(raw_path)
        if not source.exists():
            raise ValueError(f"operator package file not found: {source}")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name).strip("._-")
        safe_name = safe_name or "resource"
        if safe_name in used_names:
            raise ValueError(f"duplicate operator package resource name: {safe_name}")
        used_names.add(safe_name)
        dest = resources_dir / safe_name
        if source.is_dir():
            shutil.copytree(source, dest, symlinks=True)
            copied.append(f"_resources/{safe_name}/")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(f"_resources/{safe_name}")
    return copied


def _manifest_entries(adapted_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries_by_name: dict[str, dict[str, Any]] = {}
    for per_file in adapted_manifest.get("per_file", []):
        for entry in per_file.get("entries", []):
            entries_by_name.setdefault(entry["entry_name"], entry)
    return list(entries_by_name.values())


_QUOTED_INCLUDE_RE = re.compile(r'(?m)^(\s*#\s*include\s*)"([^"]+)"')


def _aggregate_csrc_support_files(source_csrc_dir: Path) -> list[Path]:
    support_files = []
    for source_path in sorted(source_csrc_dir.rglob("*")):
        if not source_path.is_file():
            continue
        is_top_level_asc = (
            len(source_path.relative_to(source_csrc_dir).parts) == 1
            and source_path.suffix == ".asc"
        )
        if is_top_level_asc:
            continue
        if _is_aclgraph_pybind_source_name(source_path.name):
            continue
        support_files.append(source_path)
    return support_files


def _rewrite_aggregate_support_includes(
    source_text: str, support_rel_paths: set[str], namespace: str
) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        include_path = Path(match.group(2)).as_posix()
        if include_path not in support_rel_paths:
            return match.group(0)
        return f'{prefix}"_support/{namespace}/{include_path}"'

    return _QUOTED_INCLUDE_RE.sub(replace, source_text)


def _copy_aggregate_csrc_support(
    source_csrc_dir: Path, dest_csrc_dir: Path, namespace: str
) -> list[str]:
    copied: list[str] = []
    support_root = dest_csrc_dir / "_support" / namespace
    for source_path in _aggregate_csrc_support_files(source_csrc_dir):
        rel = source_path.relative_to(source_csrc_dir)
        dest_path = support_root / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            if dest_path.read_bytes() != source_path.read_bytes():
                raise ValueError(
                    f"conflicting aggregate csrc support file: {namespace}/{rel.as_posix()}"
                )
            continue
        shutil.copy2(source_path, dest_path)
        copied.append(f"operator-sk-adapted/csrc/_support/{namespace}/{rel.as_posix()}")
    return copied


def aggregate_aclgraph_adapted_trees(
    adapted_roots: list[Path],
    output_root: Path,
    *,
    package_name: str = "op_extension",
    package_version: str = "0.1.0",
    operator_build_config: Path | None = None,
) -> dict[str, Any]:
    """Render one aclgraph-canonical package tree from multiple adapted outputs."""
    if not adapted_roots:
        raise ValueError("at least one adapted output directory is required")
    build_config = _load_operator_build_config(operator_build_config)
    output_root.mkdir(parents=True, exist_ok=True)
    adapted_dir = output_root / "operator-sk-adapted"
    if adapted_dir.exists():
        raise ValueError(f"aggregate adapted output already exists: {adapted_dir}")
    csrc_dir = adapted_dir / "csrc"
    csrc_dir.mkdir(parents=True)

    pybind_entries_by_name: dict[str, dict[str, Any]] = {}
    aggregate_per_file: list[dict[str, Any]] = []
    asset_paths: list[str] = []
    target_chips: list[str] = []
    canonical_written: list[str] = []
    used_csrc_names: set[str] = set()
    name_resolution_payload: dict[str, Any] | None = None

    for adapted_index, adapted_root in enumerate(adapted_roots):
        manifest_path = adapted_root / "operator-sk-adapted.json"
        source_dir = adapted_root / "operator-sk-adapted"
        if not manifest_path.is_file():
            raise ValueError(f"operator-sk-adapted.json not found in {adapted_root}")
        if not source_dir.is_dir():
            raise ValueError(
                f"operator-sk-adapted directory not found in {adapted_root}"
            )
        adapted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if name_resolution_payload is None and isinstance(
            adapted_manifest.get("name_resolution"), dict
        ):
            payload = adapted_manifest["name_resolution"]
            if payload.get("policy") not in {None, "", "none"}:
                name_resolution_payload = payload
        if adapted_manifest.get("status") != "completed":
            raise ValueError(f"adapted output is not completed: {adapted_root}")
        if adapted_manifest.get("pybind_layout") != "aclgraph-canonical":
            raise ValueError(
                f"adapted output is not aclgraph-canonical: {adapted_root}"
            )
        entries = _manifest_entries(adapted_manifest)
        if not entries:
            raise ValueError(f"adapted output has no kernel entries: {adapted_root}")
        for entry in entries:
            name = entry["entry_name"]
            if name in pybind_entries_by_name:
                raise ValueError(
                    f"duplicate kernel entry across adapted outputs: {name}"
                )
            pybind_entries_by_name[name] = entry
        asset_path = str(adapted_manifest.get("asset_path", adapted_root))
        asset_paths.append(asset_path)
        if adapted_manifest.get("target_chips"):
            target_chips.append(str(adapted_manifest.get("target_chips")))
        root_label = _safe_identifier(Path(asset_path).name, "asset")
        entry_label = _safe_identifier(
            str(entries[0].get("entry_name", root_label)), root_label
        )
        support_namespace = f"{entry_label}_{adapted_index}"
        source_csrc_dir = source_dir / "csrc"
        support_rel_paths = {
            path.relative_to(source_csrc_dir).as_posix()
            for path in _aggregate_csrc_support_files(source_csrc_dir)
        }

        for source_path in sorted(source_csrc_dir.glob("*.asc")):
            if _is_aclgraph_pybind_source_name(source_path.name):
                continue
            final_name = source_path.name
            if final_name in used_csrc_names:
                final_name = (
                    f"{source_path.stem}_{len(used_csrc_names)}{source_path.suffix}"
                )
            if final_name in used_csrc_names:
                raise ValueError(f"duplicate aggregate csrc name: {final_name}")
            used_csrc_names.add(final_name)
            source_text = source_path.read_text(encoding="utf-8", errors="replace")
            rendered_text = _rewrite_aggregate_support_includes(
                source_text, support_rel_paths, support_namespace
            )
            (csrc_dir / final_name).write_text(rendered_text, encoding="utf-8")
            for entry in entries:
                if (
                    entry.get("csrc_file") == source_path.name
                    or entry["entry_name"] == source_path.stem
                ):
                    entry["csrc_file"] = final_name
            canonical_written.append(f"operator-sk-adapted/csrc/{final_name}")
        canonical_written.extend(
            _copy_aggregate_csrc_support(source_csrc_dir, csrc_dir, support_namespace)
        )
        for entry in entries:
            wrapper = _runtime_wrapper_contract(entry)
            wrapper_csrc_file = str(wrapper.get("csrc_file") or "").strip()
            if not wrapper_csrc_file:
                continue
            wrapper_rel = Path(wrapper_csrc_file)
            wrapper_source = source_csrc_dir / wrapper_rel
            if not wrapper_source.is_file():
                wrapper_source = source_csrc_dir / wrapper_rel.name
            if wrapper_source.is_file():
                support_rel = wrapper_source.relative_to(source_csrc_dir).as_posix()
                wrapper["csrc_file"] = f"_support/{support_namespace}/{support_rel}"
                entry["runtime_wrapper"] = wrapper
                if isinstance(entry.get("io_contract"), dict):
                    entry["io_contract"]["runtime_wrapper"] = wrapper

        for per_file in adapted_manifest.get("per_file", []):
            if not per_file.get("entries"):
                continue
            aggregate_per_file.append(
                {
                    **per_file,
                    "file": f"{root_label}/{per_file.get('file', '')}".rstrip("/"),
                    "source_asset": str(
                        adapted_manifest.get("asset_path", adapted_root)
                    ),
                }
            )

    pybind_entries = list(pybind_entries_by_name.values())
    module_name = f"{_safe_identifier(package_name, 'op_extension')}_lib"
    (csrc_dir / "pybind11.asc").write_text(
        render_pybind11_asc(module_name, pybind_entries), encoding="utf-8"
    )
    canonical_written.append("operator-sk-adapted/csrc/pybind11.asc")
    for entry in pybind_entries:
        pybind_name = _aclgraph_entry_pybind_name(entry)
        (csrc_dir / pybind_name).write_text(
            render_pybind11_entry_asc(entry), encoding="utf-8"
        )
        canonical_written.append(f"operator-sk-adapted/csrc/{pybind_name}")

    package_dir_name = _safe_identifier(package_name, "op_extension")
    op_extension = adapted_dir / package_dir_name
    op_extension.mkdir(parents=True)
    copied_package_files = _copy_operator_package_files(op_extension, build_config)
    package_data = ["_name_resolution.json"]
    if copied_package_files:
        package_data.append("_resources/**/*")
    (op_extension / "__init__.py").write_text(
        render_op_extension_init_py(module_name, pybind_entries), encoding="utf-8"
    )
    (op_extension / "_arch_selector.py").write_text(
        render_arch_selector_py(module_name, pybind_entries), encoding="utf-8"
    )
    (op_extension / "_torch_library.py").write_text(
        render_torch_library_py(module_name, pybind_entries), encoding="utf-8"
    )
    if name_resolution_payload is not None:
        (op_extension / "_name_resolution.json").write_text(
            json.dumps(name_resolution_payload, indent=2), encoding="utf-8"
        )
    (adapted_dir / "setup.py").write_text(
        render_setup_py_aclgraph_style(
            module_name,
            pybind_entries,
            package_name=package_name,
            package_version=package_version,
            operator_build_config=build_config,
            package_data=package_data,
        ),
        encoding="utf-8",
    )
    canonical_written.extend(
        [
            f"operator-sk-adapted/{package_dir_name}/__init__.py",
            f"operator-sk-adapted/{package_dir_name}/_arch_selector.py",
            f"operator-sk-adapted/{package_dir_name}/_torch_library.py",
            *(
                [f"operator-sk-adapted/{package_dir_name}/_name_resolution.json"]
                if name_resolution_payload is not None
                else []
            ),
            *[
                f"operator-sk-adapted/{package_dir_name}/{path}"
                for path in copied_package_files
            ],
            "operator-sk-adapted/setup.py",
        ]
    )

    manifest = {
        "status": "completed",
        "aggregate": True,
        "asset_paths": asset_paths,
        "adapted_dir": str(adapted_dir),
        "package_name": package_name,
        "python_package": package_dir_name,
        "package_version": package_version,
        "pybind_layout": "aclgraph-canonical",
        "pybind_module": module_name,
        "canonical_written_files": canonical_written,
        "target_chips": ",".join(dict.fromkeys(target_chips)),
        "operator_build_config": build_config,
        "operator_package_files": copied_package_files,
        "name_resolution": name_resolution_payload
        or {"policy": "none", "renamed_entry_count": 0, "resolutions": []},
        "per_file": aggregate_per_file,
        "ignored_support_files": [],
        "escalations": [],
    }
    (output_root / "operator-sk-adapted.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


# ---------- Template loader (YAML) ----------


def _load_yaml_safe(text: str) -> Any:
    """Minimal YAML loader sufficient for our template files.

    We avoid taking on PyYAML as a hard dependency; templates are deliberately
    simple (string scalars + lists + nested dicts). If a richer file appears,
    callers should install PyYAML and switch to yaml.safe_load.
    """
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:  # pragma: no cover - fallback parser
        pass

    # Tiny indent-aware parser supporting the subset our templates use:
    # - top-level scalar key: value
    # - top-level list:   - item
    # - block scalar:    key: | + indented content
    # This is hand-rolled and intentionally limited.
    raise RuntimeError(
        "PyYAML not installed and built-in fallback insufficient; install pyyaml"
    )


def load_template(template_path: Path) -> dict[str, Any]:
    """Load a single template YAML file."""
    text = template_path.read_text(encoding="utf-8")
    data = _load_yaml_safe(text)
    if not isinstance(data, dict):
        raise ValueError(f"template {template_path} must be a mapping at top level")
    required = {"id", "description", "parameters", "files"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"template {template_path} missing keys: {sorted(missing)}")
    return data


def list_templates(templates_dir: Path) -> list[dict[str, Any]]:
    if not templates_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(templates_dir.glob("*.yaml")):
        try:
            data = load_template(path)
        except Exception as exc:  # pragma: no cover - surfaced via CLI
            items.append({"id": path.stem, "path": str(path), "error": str(exc)})
            continue
        items.append(
            {
                "id": data["id"],
                "description": data["description"],
                "path": str(path),
                "parameters": data["parameters"],
                "file_count": len(data["files"]),
            }
        )
    return items


def _coerce_param_value(spec: dict, raw: Any) -> Any:
    kind = spec.get("kind", "string")
    if kind == "int":
        return int(raw)
    if kind == "choice":
        allowed = spec.get("allowed") or []
        if raw not in allowed:
            raise ValueError(
                f"value {raw!r} not in allowed {allowed!r} for parameter {spec.get('name')!r}"
            )
        return raw
    return str(raw)


def resolve_template_params(
    template: dict, overrides: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seen: set[str] = set()
    for spec in template["parameters"]:
        name = spec["name"]
        seen.add(name)
        raw = overrides.get(name, spec.get("default"))
        if raw is None:
            raise ValueError(
                f"parameter {name!r} missing (no value supplied and no default)"
            )
        out[name] = _coerce_param_value(spec, raw)
    extras = set(overrides) - seen
    if extras:
        raise ValueError(f"unknown template parameters: {sorted(extras)}")
    return out


_DTYPE_TO_CTYPE = {
    "float16": "half",
    "float32": "float",
    "int32": "int32_t",
    "int64": "int64_t",
}


def render_template_files(
    template: dict, params: dict[str, Any]
) -> list[tuple[str, str]]:
    """Render every file entry in the template, returning [(rel_path, content), ...].

    Uses string.Template's safe_substitute ($key syntax) to avoid clashing with
    the curly braces that appear in every C/C++ kernel body.
    """
    from string import Template

    ctx = {str(k): str(v) for k, v in params.items()}
    if "dtype" in ctx:
        ctx.setdefault("ctype", _DTYPE_TO_CTYPE.get(ctx["dtype"], ctx["dtype"]))
    ctx.setdefault("id", str(template["id"]))
    rendered: list[tuple[str, str]] = []
    for file_entry in template["files"]:
        path_str = Template(file_entry["path"]).safe_substitute(ctx)
        body = Template(file_entry["template"]).safe_substitute(ctx)
        rendered.append((path_str, body))
    return rendered


# ---------- Remediation applier ----------

AUTO_REMEDIATION_KINDS = frozenset(
    {"remove-line-containing", "rename-symbol", "add-include", "replace-pattern"}
)
INLINE_CLEAN_SCHEMA_VERSION = 1
INLINE_CLEAN_CANONICAL_OWNER = "sk-operator-codegen"
_INLINE_LEGACY_SYS_ARGS = (
    ("skBlockNum", "skNumBlocks"),
    ("SkGetBlockNum", "SkGetNumBlocks"),
)


def _inline_pre_adapt_findings(asset_dir: Path) -> list[dict]:
    findings: list[dict] = []
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file() or path.suffix not in {
            ".asc",
            ".cpp",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
        }:
            continue
        rel = path.relative_to(asset_dir).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for legacy, modern in _INLINE_LEGACY_SYS_ARGS:
            if not re.search(rf"\b{legacy}\b", text):
                continue
            findings.append(
                {
                    "finding_id": f"codegen.inline-sys-args-api-current:{rel}:{legacy}",
                    "rule_id": "codegen.inline-sys-args-api-current",
                    "severity": "blocker",
                    "category": "spec",
                    "actionable_by": ["codegen.apply-remediation"],
                    "remediation_hint": {
                        "kind": "rename-symbol",
                        "target_file": rel,
                        "old_value": legacy,
                        "new_value": modern,
                    },
                    "evidence": [legacy],
                    "message": f"legacy SkSystemArgs API {legacy!r}; rename to {modern!r}.",
                }
            )
    return findings


def apply_remediation(
    asset_dir: Path,
    findings: list[dict],
) -> list[dict]:
    """Apply auto-remediable findings to source files under asset_dir.

    Returns one result dict per finding:
        {"finding_id": ..., "status": "applied|escalated|skipped|failed", "reason": ..., "diff_summary": ...}
    """
    results: list[dict] = []
    for finding in findings:
        hint = finding.get("remediation_hint", {})
        kind = hint.get("kind")
        finding_id = finding.get("finding_id", "<unknown>")
        actionable = finding.get("actionable_by", [])

        if (
            "codegen.apply-remediation" not in actionable
            or kind not in AUTO_REMEDIATION_KINDS
        ):
            results.append(
                {
                    "finding_id": finding_id,
                    "status": "escalated",
                    "reason": "not_auto_remediable",
                    "diff_summary": "",
                }
            )
            continue

        target_rel = hint.get("target_file") or ""
        if not target_rel:
            results.append(
                {
                    "finding_id": finding_id,
                    "status": "failed",
                    "reason": "missing_target_file",
                    "diff_summary": "",
                }
            )
            continue
        target = (asset_dir / target_rel).resolve()
        try:
            target.relative_to(asset_dir.resolve())
        except ValueError:
            results.append(
                {
                    "finding_id": finding_id,
                    "status": "failed",
                    "reason": "target_outside_asset",
                    "diff_summary": "",
                }
            )
            continue
        if not target.exists() or not target.is_file():
            results.append(
                {
                    "finding_id": finding_id,
                    "status": "failed",
                    "reason": "target_not_found",
                    "diff_summary": "",
                }
            )
            continue

        text = target.read_text(encoding="utf-8")
        before_size = len(text)
        if kind == "remove-line-containing":
            needle = hint.get("old_value", "")
            kept_lines = []
            for line in text.splitlines(keepends=False):
                if needle not in line:
                    kept_lines.append(line)
            new_text = "\n".join(kept_lines)
            if text.endswith("\n"):
                new_text += "\n"
        elif kind == "rename-symbol":
            old = hint.get("old_value", "")
            new = hint.get("new_value", "")
            if not old:
                results.append(
                    {
                        "finding_id": finding_id,
                        "status": "failed",
                        "reason": "missing_old_value",
                        "diff_summary": "",
                    }
                )
                continue
            # Use word-boundary replacement for safety.
            new_text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        elif kind == "add-include":
            include_line = hint.get("new_value", "")
            if include_line and include_line not in text:
                # Insert after the last #include line if present, else at top.
                lines = text.splitlines(keepends=True)
                insert_at = 0
                for i, ln in enumerate(lines):
                    if ln.lstrip().startswith("#include"):
                        insert_at = i + 1
                lines.insert(
                    insert_at,
                    include_line + ("\n" if not include_line.endswith("\n") else ""),
                )
                new_text = "".join(lines)
            else:
                new_text = text
        elif kind == "replace-pattern":
            pattern = hint.get("old_value", "")
            replacement = hint.get("new_value", "")
            if not pattern:
                results.append(
                    {
                        "finding_id": finding_id,
                        "status": "failed",
                        "reason": "missing_pattern",
                        "diff_summary": "",
                    }
                )
                continue
            try:
                new_text = re.sub(pattern, replacement, text)
            except re.error as exc:
                results.append(
                    {
                        "finding_id": finding_id,
                        "status": "failed",
                        "reason": f"invalid_regex:{exc}",
                        "diff_summary": "",
                    }
                )
                continue
        else:  # pragma: no cover - guarded above
            new_text = text

        if new_text == text:
            results.append(
                {
                    "finding_id": finding_id,
                    "status": "skipped",
                    "reason": "no_change",
                    "diff_summary": "",
                }
            )
            continue
        target.write_text(new_text, encoding="utf-8")
        results.append(
            {
                "finding_id": finding_id,
                "status": "applied",
                "reason": kind,
                "diff_summary": f"size {before_size} -> {len(new_text)}",
            }
        )
    return results


def _run_spec_clean_loop_inline(
    source_text: str,
    *,
    rel_path: str = "input.asc",
    max_rounds: int = 3,
) -> tuple[str, list[dict], list[dict]]:
    """Run codegen-owned, auto-remediable pre-adapt rules before SK adaptation."""
    inline_remediations: list[dict] = []
    escalations: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="sk-codegen-inline-clean-") as tmp:
        tmp_root = Path(tmp)
        asset_dir = tmp_root / "asset"
        source_path = asset_dir / rel_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source_text, encoding="utf-8")

        for round_index in range(max_rounds):
            findings = _inline_pre_adapt_findings(asset_dir)
            blockers = []
            auto_findings = []
            for finding in findings:
                if finding.get("severity") == "blocker":
                    blockers.append(finding)
                actionable_by = finding.get("actionable_by", [])
                remediation_kind = finding.get("remediation_hint", {}).get("kind")
                if (
                    "codegen.apply-remediation" in actionable_by
                    and remediation_kind in AUTO_REMEDIATION_KINDS
                ):
                    auto_findings.append(finding)
            if not blockers and not auto_findings:
                return source_path.read_text(encoding="utf-8"), inline_remediations, []

            remediable = auto_findings if auto_findings else blockers
            results = apply_remediation(asset_dir, remediable)
            inline_remediations.append(
                {
                    "round": round_index,
                    "findings": remediable,
                    "results": results,
                }
            )
            failed = [r for r in results if r.get("status") in ("failed", "escalated")]
            applied = [r for r in results if r.get("status") == "applied"]
            if not blockers and not applied:
                return source_path.read_text(encoding="utf-8"), inline_remediations, []
            if failed or (blockers and not applied):
                escalations.extend(
                    _human_finding(
                        f.get("finding_id", "codegen.inline-spec-clean-unresolved"),
                        f.get(
                            "message",
                            "inline spec-clean could not auto-remediate a blocker finding.",
                        ),
                        f.get("evidence", []),
                    )
                    for f in blockers
                )
                break

        if not escalations:
            escalations.append(
                _human_finding(
                    "codegen.inline-spec-clean-unresolved",
                    f"inline spec-clean still had blocker findings after {max_rounds} rounds.",
                )
            )
        return source_path.read_text(encoding="utf-8"), inline_remediations, escalations
