#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import re
import csv
import sys
import logging
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')

sim_map = {
    "AIC_MTE1": "aic_mte1_ratio",
    "AIC_MTE2": "aic_mte2_ratio",
    "AIC_FIXPIPE": "aic_fixpipe_ratio",
    "AIC_MAC": "aic_mac_ratio",
    "AICORE_VEC": "aiv_vec_ratio",
    "AIV_MTE2": "aiv_mte2_ratio",
    "AIV_MTE3": "aiv_mte3_ratio",
    "AIV_VEC": "aiv_vec_ratio"
}

cycle_ref = {
    "AIC_MTE1": "aic_total_cycles",
    "AIC_MTE2": "aic_total_cycles",
    "AIC_FIXPIPE": "aic_total_cycles",
    "AIC_MAC": "aic_total_cycles",
    "AICORE_VEC": "aiv_total_cycles",
    "AIV_MTE2": "aiv_total_cycles",
    "AIV_MTE3": "aiv_total_cycles",
    "AIV_VEC": "aiv_total_cycles"
}


def get_case(info):
    res = dict()
    pattern_getcase = (r"\[INFO\][^\n]*\[([^\n\[\]]*?)\]Start GetTiling\."
                       r"(.*?)(\[INFO\][^\n]*\[([^\n\[\]]*?)\]End GetTiling\.|\[ERROR\])")
    matches = re.findall(pattern_getcase, info, re.DOTALL)
    for match in matches:
        op_name = match[0]
        if op_name not in res:
            res[op_name] = []
        res[op_name].append(match[1])
    return res


def get_sched_group(op_name, infos):
    res = dict()
    pattern_getgraph = (r"\[INFO\][^\n]*\[{}\]Start tiling for sched group (.*?)\."
                        r"(.*?)\[INFO\][^\n]*\[{}\]End tiling for sched group\.".format(op_name, op_name))
    matches = re.findall(pattern_getgraph, infos, re.DOTALL)
    if len(matches) > 0:
        for match in matches:
            res[match[0]] = match[1]
    else:
        res["uniq_group"] = infos
    return res


def get_sched_result(op_name, infos):
    res = dict()
    pattern_getcase = (r"\[INFO\][^\n]*\[{}\]\[PROF\]Among all schedule results, "
                       r"(.*?) is the best choice\.".format(op_name))
    match = re.findall(pattern_getcase, infos, re.DOTALL)
    if len(match) > 0:
        return match[0][0]
    else:
        return None


def is_sched_result(sched_group, sched_result):
    if sched_result is None:
        return True
    return sched_group.startswith("ScheduleResult" + sched_result)


def get_graph(op_name, infos):
    res = dict()
    pattern_getgraph = (r"\[DEBUG\][^\n]*\[{}\]Calculating the tiling data for tilingCaseId (\d*?)\."
                        r"(.*?)\[DEBUG\][^\n]*\[{}\]Finish calculating the tiling data for "
                        r"tilingCaseId (\d*?)\.".format(op_name, op_name))
    matches = re.findall(pattern_getgraph, infos, re.DOTALL)
    res_getgraph = (r"\[INFO\][^\n]*\[{}\]\[PROF\]Among the templates, tiling case (\d*?) of "
                    r"(.*?) is the best choice\.".format(op_name))
    result = re.findall(res_getgraph, infos, re.DOTALL)
    for match in matches:
        if (match[0] == match[2]):
            case_id = int(match[0])
            if case_id not in res:
                res[case_id] = [case_id == int(result[0][0]), match[1]]
    return res


def pass_valid_check(op_name, infos):
    pattern_keylog = r"\[DEBUG\][^\n]*\[{}\]Execute DoTiling\.".format(op_name)
    if re.search(pattern_keylog, infos):
        return True
    else:
        return False


def check_context(op_name, infos):
    pattern_keylog = r"\[INFO\][^\n]*\[{}\]Start context tiling\.".format(op_name)
    if re.search(pattern_keylog, infos):
        return True
    else:
        return False


def get_input_vars(op_name, tiling_key, infos):
    input_vars = dict()
    pattern_getinput = (r"\[DEBUG\][^\n]*\[{}\]Start setting axis size for {}\."
                        r"(.*?)\[DEBUG\][^\n]*\[{}\]End setting axis size for "
                        r"{}\.".format(op_name, tiling_key, op_name, tiling_key))
    matches = re.findall(pattern_getinput, infos, re.DOTALL)
    initiate_info = matches[0]
    pattern_getvar = r"Initiate (.*?) to (\d*?)\."
    var_info = re.findall(pattern_getvar, initiate_info)
    for var in var_info:
        input_vars[var[0]] = int(var[1])
    return input_vars


def find_iter_num(op_name, infos):
    pattern_keylog = r"iter : (\d+?)"
    matches = re.findall(pattern_keylog, infos)
    return len(matches)


def find_solution(info):
    pattern_keylog = r"Feasible solution"
    if re.search(pattern_keylog, info):
        return True
    else:
        return False


def obtain_error_log(op_name, info):
    pattern_keylog = r"\[WARNING\][^\n]*\[{}\](.*?)\n".format(op_name)
    matches = re.findall(pattern_keylog, info)
    return matches[0]


def get_hardware_info(op_name, info):
    params = dict()
    pattern_getstatus = r"\[DEBUG\][^\n]*\[{}\]Set hardware params. (.*?)\n".format(op_name)
    pattern_getparams = r"(.*?) = (\d*?)\."
    matches = re.findall(pattern_getstatus, info)
    if len(matches) > 0:
        status = re.findall(pattern_getparams, matches[0])
        for item in status:
            params[item[0]] = int(item[1])
    return params


def _calc_real_cost(df, op_name, cost_key, tiling_res):
    real_cost = []
    for case in df[op_name]:
        real_cost.append(case[sim_map[cost_key]] * case[cycle_ref[cost_key]])
    return real_cost


def solution_analysis(op_name, tiling_key, tiling_info, tiling_res, df):
    tiling_data = dict()
    sim_cost = dict()
    sim_expr = dict()
    real_cost = dict()
    occupy = dict()
    pattern_getsolution = (r"\[DEBUG\][^\n]*\[{}\]Filling tilingdata for case{}\.\n(.*?)"
                           r"\[DEBUG\][^\n]*\[{}\]Objective value for case{} is "
                           r"(\d*?\.*?\d*?)\.\n".format(op_name, tiling_key, op_name, tiling_key))
    pattern_getstatus = r"\[DEBUG\][^\n]*\[{}\](.*?) = (\d*?\.*?\d*?)\n".format(op_name)
    pattern_split = r"\[DEBUG\][^\n]*\[{}\]Simulate the cost\.\n".format(op_name)
    pattern_expr = r"\[DEBUG\][^\n]*\[{}\]The expression of (.*?) is (.*?)\n".format(op_name)
    matches = re.findall(pattern_getsolution, tiling_info, re.DOTALL)[0]
    split_strs = re.split(pattern_split, matches[0], re.DOTALL)
    status = re.findall(pattern_getstatus, split_strs[0])
    for item in status:
        occupy[item[0]] = float(item[1])
    status = re.findall(pattern_getstatus, split_strs[1])
    for item in status:
        sim_cost[item[0]] = float(item[1])
        if df is not None and tiling_res["best_case"]:
            real_cost[item[0]] = _calc_real_cost(df, op_name, item[0], tiling_res)
    exprs = re.findall(pattern_expr, split_strs[1])
    for item in exprs:
        sim_expr[item[0]] = item[1]
    pattern_res = (r"\[DEBUG\][^\n]*\[{}\]The output of the solver for tilingCaseId case{} is:\n"
                   r"(.*?)\[DEBUG\][^\n]*\[{}\]The solver executed "
                   r"successfully\.".format(op_name, tiling_key, op_name))
    res_info = re.findall(pattern_res, tiling_info, re.DOTALL)[0]
    val_res = re.findall(pattern_getstatus, res_info)
    for item in val_res:
        tiling_data[item[0]] = int(item[1])
    tiling_res["obj"] = float(matches[1])
    tiling_res["solution"] = tiling_data
    tiling_res["occupy"] = occupy
    tiling_res["sim_cost"] = sim_cost
    tiling_res["sim_expr"] = sim_expr
    tiling_res["real_cost"] = real_cost


def format_number(num, simplify=False):
    ret_num = num
    if isinstance(num, float):
        if int(num) == num:
            ret_num = int(num)
        else:
            ret_num = round(num, 2)
    if simplify:
        if (ret_num >= 1024 * 1024):
            return str(round(ret_num / 1024 / 1024, 2)) + "m"
        if (ret_num >= 1024):
            return str(round(ret_num / 1024, 2)) + "k"
    return str(ret_num)


def print_dict(used_dict, compare_dict=None, header=None, simplify=False):
    logging.info("------------------------------------------------")
    if header is not None and len(header) > 0:
        header_str = "      "
        for header_item in header:
            header_str += "{:<15}".format(header_item)
        logging.info(header_str)
        logging.info("------------------------------------------------")
    for key, value in used_dict.items():
        if compare_dict is not None:
            logging.info("      {:<15}{:<15}{:<15}".format(
                key, format_number(value, simplify), format_number(compare_dict[key], simplify)))
        else:
            logging.info("      {:<15}{:<15}".format(key, format_number(value, simplify)))
    logging.info("------------------------------------------------")
    logging.info("")


def _print_tiling_solution(tiling_res, has_summary):
    logging.info("    Solution:")
    print_dict(tiling_res["solution"])
    logging.info("    Estimate:")
    if has_summary:
        print_dict(tiling_res["sim_cost"],
                   compare_dict=tiling_res["real_cost"],
                   header=["pipe", "sim_cost", "real_cost"])
    else:
        print_dict(tiling_res["sim_cost"], header=["pipe", "sim_cost"])
    print_dict(tiling_res["occupy"], header=["loc", "occupy"], simplify=True)


def _print_tiling_res(tiling_res, has_summary):
    logging.info("   tilingCaseId: {:<20}".format(tiling_res["tilingCaseId"]))
    if not tiling_res["pass_check"]:
        logging.info("    Status: {}".format(tiling_res["error_msg"]))
        return
    if "input_var" in tiling_res:
        logging.info("    input vars:")
        print_dict(tiling_res["input_var"])
    logging.info("    hardware params:")
    print_dict(tiling_res["hardware_param"], simplify=True)
    logging.info("    exe_iter: {:<20}".format(tiling_res["exec_iter"]))
    if tiling_res["has_solution"]:
        _print_tiling_solution(tiling_res, has_summary)
    else:
        logging.info("    Status: {}".format(tiling_res["error_msg"]))


def _print_sched_group(sched_group, sched_info, has_summary):
    if sched_group != "uniq_group":
        sched_str = "  Sched Group: {}".format(sched_group)
        if sched_info[0]:
            sched_str += "(chosen)"
        logging.info(sched_str)
    for tiling_res in sched_info[1].values():
        _print_tiling_res(tiling_res, has_summary)


def display_res(tiling_ret, has_summary):
    for op_name, op_info in tiling_ret.items():
        logging.info("Op Name: {:<20}".format(op_name))
        for case_id, case_info in op_info.items():
            logging.info(" Case: {:<20}".format(case_id))
            for sched_group, sched_info in case_info.items():
                _print_sched_group(sched_group, sched_info, has_summary)
    logging.info("")


def _analyze_tiling_cases(op_name, tiling_infos, df):
    sched_ret = dict()
    for tiling_case, tiling_infomation in tiling_infos.items():
        cur_case = "tilingCase{}".format(tiling_case)
        tiling_res = {}
        best_case, tiling_info = tiling_infomation
        tiling_res["best_case"] = best_case
        tiling_res["tilingCaseId"] = tiling_case
        tiling_res["error_msg"] = ""
        tiling_res["hardware_param"] = get_hardware_info(op_name, tiling_info)
        tiling_res["pass_check"] = True
        if (check_context(op_name, tiling_info)):
            tiling_res["input_var"] = get_input_vars(op_name, tiling_case, tiling_info)
        tiling_res["exec_iter"] = find_iter_num(op_name, tiling_info)
        tiling_res["has_solution"] = find_solution(tiling_info)
        if (tiling_res["has_solution"]):
            solution_analysis(op_name, tiling_case, tiling_info, tiling_res, df)
        else:
            tiling_res["error_msg"] = obtain_error_log(op_name, tiling_info)
        sched_ret[cur_case] = tiling_res
    return sched_ret


def analysis_info(info, df):
    case_infos = get_case(info)
    op_ret = dict()
    for op_name, case_list in case_infos.items():
        input_ret = dict()
        for i, case_info in enumerate(case_list):
            cur_input = "case{}".format(i + 1)
            case_ret = dict()
            sched_result = get_sched_result(op_name, case_info)
            sched_groups = get_sched_group(op_name, case_info)
            for sched_group, _ in sched_groups.items():
                tiling_infos = get_graph(op_name, case_info)
                sched_ret = _analyze_tiling_cases(op_name, tiling_infos, df)
                case_ret[sched_group] = [is_sched_result(sched_group, sched_result), sched_ret]
            input_ret[cur_input] = case_ret
        op_ret[op_name] = input_ret
    return op_ret


def _parse_csv_row(row, header):
    col_info = dict()
    for key_name in sim_map.values():
        col_info[key_name] = row[header[key_name]]
    for key_name in cycle_ref.values():
        col_info[key_name] = row[header[key_name]]
    return col_info


def _parse_csv_header(row):
    header = dict()
    for j, col_name in enumerate(row):
        header[col_name] = j
    return header


def _append_row_to_dataframe(data_frame, row, header):
    op_name = row[header["op_name"]]
    if op_name not in data_frame:
        data_frame[op_name] = []
    data_frame[op_name].append(_parse_csv_row(row, header))


def _parse_csv_to_dataframe(profiler_path):
    data_frame = dict()
    header = dict()
    with open(profiler_path, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                header = _parse_csv_header(row)
            else:
                _append_row_to_dataframe(data_frame, row, header)
    return data_frame


def get_data_frame(folder_path):
    csv_path = os.path.join(folder_path, "mindstudio_profiler_output")
    if os.path.exists(csv_path):
        for file in os.listdir(csv_path):
            if file.endswith("op_summary"):
                profiler_path = os.path.join(csv_path, file)
                return _parse_csv_to_dataframe(profiler_path)
    return None


def _build_tiling_info_str(tiling_res):
    tiling_info = ""
    for key, value in tiling_res["solution"].items():
        if len(tiling_info) > 0:
            tiling_info += ","
        tiling_info += "{} = {}".format(key, value)
    return tiling_info


def _build_pipe_costs(tiling_res, pipetype, has_summary):
    costs = []
    if pipetype in tiling_res["sim_cost"]:
        costs += [tiling_res["sim_cost"][pipetype], tiling_res["sim_expr"][pipetype]]
    else:
        costs += [0, 0]
    if has_summary and pipetype in tiling_res["real_cost"]:
        costs.append(tiling_res["real_cost"][pipetype])
    elif has_summary:
        costs.append(0)
    return costs


def _build_row_base(op_name, cur_sched, tiling_res):
    col_info = [op_name, cur_sched, tiling_res["tilingCaseId"]]
    col_info.append(_build_tiling_info_str(tiling_res))
    if ("block_dim" in tiling_res["occupy"]):
        col_info.append(tiling_res["occupy"]["block_dim"])
    else:
        col_info.append(0)
    return col_info


def _build_summary_rows(col_info, tiling_res):
    rows = []
    test_num = len(list(tiling_res["real_cost"].values())[0])
    for i in range(test_num):
        cur_col_info = list(col_info)
        for pipetype in sim_map.keys():
            pipe_costs = _build_pipe_costs(tiling_res, pipetype, True)
            cur_col_info.append(pipe_costs[0])
            cur_col_info.append(pipe_costs[1])
            if i < len(tiling_res["real_cost"].get(pipetype, [])):
                cur_col_info.append(tiling_res["real_cost"][pipetype][i])
            else:
                cur_col_info.append(0)
        rows.append(cur_col_info)
    return rows


def _build_no_summary_row(col_info, tiling_res):
    for pipetype in sim_map.keys():
        col_info += _build_pipe_costs(tiling_res, pipetype, False)
    return col_info


def _collect_sched_rows(op_name, cur_sched, sched_ret, has_summary):
    rows = []
    if not sched_ret[0]:
        return rows
    for _, tiling_res in sched_ret[1].items():
        if (not tiling_res["best_case"]) or (len(tiling_res["sim_cost"]) == 0):
            continue
        col_info = _build_row_base(op_name, cur_sched, tiling_res)
        if has_summary:
            rows.extend(_build_summary_rows(col_info, tiling_res))
        else:
            rows.append(_build_no_summary_row(col_info, tiling_res))
        break
    return rows


def _collect_output_rows(op_ret, has_summary):
    rows = []
    for op_name, input_ret in op_ret.items():
        for _, case_ret in input_ret.items():
            for cur_sched, sched_ret in case_ret.items():
                rows.extend(_collect_sched_rows(op_name, cur_sched, sched_ret, has_summary))
    return rows


def _build_output_header(has_summary):
    header = ["op_name", "sched group", "tilingCaseId", "tilingData", "block_dim"]
    for pipetype in sim_map.keys():
        header += ["sim_" + pipetype, pipetype + "_expr"]
        if has_summary:
            header.append("real_" + pipetype)
    return header


def output_res(op_ret, has_summary, output_path):
    header = _build_output_header(has_summary)
    rows = _collect_output_rows(op_ret, has_summary)
    data = [header] + rows
    with open(output_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='Path to the tiling log')
    parser.add_argument('--summary_path', type=str, default='./PROF',
                        help='Path to the folder containing the profiling data')
    args = parser.parse_args()
    with open(args.log_path, 'r', encoding='utf-8') as file:
        info = file.read()
    data_frame = get_data_frame(args.summary_path)
    if info is not None:
        has_summary = (data_frame is not None)
        op_ret = analysis_info(info, data_frame)
        display_res(op_ret, has_summary)
        output_res(op_ret, has_summary, "./output.csv")
