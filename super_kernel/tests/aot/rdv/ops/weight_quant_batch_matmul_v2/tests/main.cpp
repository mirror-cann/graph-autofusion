/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "DevContext.h"
#include "OpsInterface.h"

int32_t main(int32_t argc, char *argv[])
{
    DevContext devObj(0);
    int startId = devObj.dev_ptrs_.size();
    devObj.dev_ptrs_.resize(startId + 5);
    uint16_t buf[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = 0x3c00;
    }
    for (auto i = 0; i < 5; i++) {
        CHECK_ACL(aclrtMalloc((void **)&(devObj.dev_ptrs_[startId+i]), 4096 * 4096 * 2, ACL_MEM_MALLOC_HUGE_FIRST));
        if (i != 1) {
            for (int j = 0; j < 10; j++) {
                CHECK_ACL(aclrtMemcpy((char *)devObj.dev_ptrs_[startId+i] + j * 2048, 2048, buf, 2048, ACL_MEMCPY_HOST_TO_DEVICE));
            }
        } else {
            CHECK_ACL(aclrtMemset(devObj.dev_ptrs_[startId+i], 20480, 1, 20480));
        }
    }
    // WeightQuantBatchMatmulV2TilingData tiling = {7, 1, 1, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 512, 256, 1, 1, 0, 7, 1, 0, 1, 0, 320, 256, 320, 256, 0, 8192, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {1, 8192, 256, 320, 320, 512, 256, 320, 128, 256, 64, 20, 2, 4, 1, 0, 131072, 1, 0, 393216, 131072, 0, 1, 1, 1, 1, 5, 1, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }}; // cost: 1 min time
    WeightQuantBatchMatmulV2MsdTilingData tilingMc = {1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 32, 320, 32, 320, 256, 0, 0, {1, 64, 256, 320, 320, 64, 256, 320, 64, 256, 128, 3, 3, 1, 1, 0, 0, 0, 0, 122880, 65536, 0, 1, 1, 1, 1, 3, 3, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }};
    WeightQuantBatchMatmulV2Kernel(16, devObj.stream_, (uint8_t *)devObj.dev_ptrs_[startId+0], (uint8_t *)devObj.dev_ptrs_[startId+1], (uint8_t *)devObj.dev_ptrs_[startId+2], (uint8_t *)devObj.dev_ptrs_[startId+3], nullptr, nullptr, nullptr, (uint8_t *)devObj.dev_ptrs_[startId+4], (uint8_t *)devObj.workspace_, tilingMc);
    return 0;
}
