#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------
"""
gen sk entry stub
"""

import sys
import os
import subprocess
import re


def get_file_content(file):
    data = []
    with open(file, 'rb') as fd:
        buf = fd.read()
        start = 0
        while start + 8 <= len(buf):
            value = int.from_bytes(buf[start:start + 8], byteorder='little')
            data.append(f'0x{value:016x}')
            start += 8
    return data


def gen_code(bin_file, src):
    file_size = os.path.getsize(bin_file)
    if file_size % 8 != 0:
        raise ValueError(f"Binary file {bin_file} (size: {file_size}) must be 8-byte aligned.")
    data = get_file_content(bin_file)
    formatted_data = [
        "    " + ", ".join(data[i:i + 8]) 
        for i in range(0, len(data), 8)
    ]
    data_lines = ",\n".join(formatted_data)

    code = f'''#include <acl/acl.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include "sk_log.h"

#ifdef __cplusplus
extern "C" {{
#endif

static aclrtBinHandle AscendLoadBinaryFromBuffer(const uint64_t *aicoreFileBuf, size_t fileSize)
{{
    constexpr uint32_t optLen = 2;
    aclrtBinaryLoadOption opList[optLen] = {{
        {{ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC, {{ACL_RT_BINARY_MAGIC_ELF_AICORE}}}},
        {{ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, {{/* isLazyLoad = */1}}}}
    }};
    aclrtBinaryLoadOptions opts = {{opList, optLen}};
    
    aclrtBinHandle bhdl = nullptr;
    int32_t ret = aclrtBinaryLoadFromData((const char *)aicoreFileBuf, fileSize, &opts, &bhdl);
    if (ret != 0) {{
        SK_LOGE("aclrtBinaryLoadFromData error, please check log!");
    }}
    return bhdl;
}}

static const uint64_t __aicore_file_buf__[{len(data)}] __attribute__ ((section (".sk.kernel"))) = {{
{data_lines}
}};

aclrtBinHandle AscendGetEntryBinHandle()
{{
    static thread_local aclrtBinHandle __sk_bin_handle__ = nullptr;
    if (__sk_bin_handle__ == nullptr) {{
        __sk_bin_handle__ = AscendLoadBinaryFromBuffer(__aicore_file_buf__, sizeof(__aicore_file_buf__));
    }}
    return __sk_bin_handle__;
}}

#ifdef __cplusplus
}}
#endif
'''

    try:
        with open(src, 'w+', encoding='utf-8') as fd:
            fd.write(code)
    except Exception as err:
        raise RuntimeError(f"Could not write to {src}: {err}") from err


if __name__ == '__main__':
    bin_file = sys.argv[1]
    src_file = sys.argv[2]
    gen_code(bin_file, src_file)