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
    devObj.dev_ptrs_.resize(startId+8);
    uint16_t buf[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = 0x3c00;
    }
    for (auto i = 0; i < 8; i++) {
        CHECK_ACL(aclrtMalloc((void **)&(devObj.dev_ptrs_[startId+i]), 4096 * 4096, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(devObj.dev_ptrs_[startId+i], 2048, buf, 2048, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    GMMTilingData tiling = {
        {1, 20, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
        {{32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
        {20, 32, 32, 32, 32, 32, 256, 32, 32, 256, 64, 56, 8, 1, 1, 0, 0, 0, 0, 18432, 32768, 0, 1, 1, 1, 1, 28, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    };
    int64_t shape[2] = {32, 32};
    void *dev_ptrs[8];
    for (int i = 0; i < 8; i++) {
        void *dev[1] = {devObj.dev_ptrs_[startId+i]};
        dev_ptrs[i] = create_tensor_list(2, shape, 1, dev);
    }
    GroupedMatmulKernel(20, devObj.stream_,
                        (uint8_t *)dev_ptrs[0], (uint8_t *)dev_ptrs[1], (uint8_t *)dev_ptrs[2], (uint8_t *)dev_ptrs[3],
                        (uint8_t *)dev_ptrs[4], (uint8_t *)dev_ptrs[5], (uint8_t *)dev_ptrs[6],
                        nullptr, nullptr, (uint8_t *)dev_ptrs[7],
                        (uint8_t *)devObj.workspace_, tiling);
    return 0;
}
