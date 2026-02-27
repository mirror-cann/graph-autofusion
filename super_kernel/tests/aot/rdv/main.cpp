/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "acl/acl.h"
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include "super_kernel.h"
#include "rt_intf.h"
#include "rt_sk_intf.h"
#include "DevContext.h"
#include "OpsInterface.h"

void gen_kernel_op_with_case(DevContext &devObj, int run_case){
    // Clear and explicit: case 0 means "all ops"; other cases map to a single op.
    switch (run_case) {
    case 0:
        // run all supported ops
        gen_rms_kernel_func(devObj);
        gen_grouped_matmul_func(devObj);
        gen_weight_quant_batch_matmul_v2_func(devObj);
        gen_matmul_add_func(devObj);
        gen_dequant_swiglu_quant_func(devObj);
        break;
    case 1:
        gen_rms_kernel_func(devObj);
        break;
    case 2:
        gen_grouped_matmul_func(devObj);
        break;
    case 3:
        gen_weight_quant_batch_matmul_v2_func(devObj);
        break;
    case 4:
        gen_matmul_add_func(devObj);
        break;
    case 5:
        gen_dequant_swiglu_quant_func(devObj);
        break;
    case 10:
        {
            auto layer_params_idx_size = network__memory_malloc(devObj);
            launch_network(devObj, layer_params_idx_size, 50);
        }
        break;
    default:
        // unknown code: no-op
        break;
    }
}

int32_t main(int32_t argc, char *argv[])
{
    DevContext devObj(0);
    rt_start_capture(1); // 1: dry run, not launch to device
    // parse multiple run modes from argv. argv[1..argc-1]
    printf("argc=%d\n", argc);
    for (int i = 0; i < 1; i++) {
        for (int j = 1; j < argc; ++j) {
            gen_kernel_op_with_case(devObj, atoi(argv[j]));
        }
    }

    rt_stop_capture();
    rt_sk_capture_snapshot();

    aclmdlRI sub_var1;
    aclskOptions *sub_var2 = nullptr;
    CHECK_ACL(aclskOptimize(sub_var1, sub_var2));

    rt_sk_replay();

    return 0;
}
