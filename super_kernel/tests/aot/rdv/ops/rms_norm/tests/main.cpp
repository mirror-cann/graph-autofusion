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
    devObj.dev_ptrs_.resize(startId + 4);
    uint16_t buf[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = 0x3c00 + i;
    }
    for (auto i = 0; i < 4; i++) {
        CHECK_ACL(aclrtMalloc((void **)&(devObj.dev_ptrs_[startId+i]), 4096 * 4096, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(devObj.dev_ptrs_[startId+i], 2048, buf, 2048, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    RMSNormTilingData tiling = {1, 1024, 1024, 1, 13, 13312, 0, 0, 0, 0, 0, 0, 0, 0, 1, 9.999999747378752e-06, 0.0009765625, 0, {0, 0, 0}};
    RmsNormKernel(1, devObj.stream_, (uint8_t *)devObj.dev_ptrs_[startId], (uint8_t *)devObj.dev_ptrs_[startId+1], (uint8_t *)devObj.dev_ptrs_[startId+3], (uint8_t *)devObj.dev_ptrs_[startId+2], tiling);
    return 0;
}
