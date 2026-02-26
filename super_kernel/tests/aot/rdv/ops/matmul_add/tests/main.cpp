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
    devObj.dev_ptrs_.resize(2);
    uint8_t buf[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = 1;
    }
    for (auto i = 0; i < 2; i++) {
        CHECK_ACL(aclrtMalloc((void **)&(devObj.dev_ptrs_[i]), 4096 * 4096 * 2, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemset(devObj.dev_ptrs_[i], 4096, 1, 4096));
    }
    MatmulAdd(20, devObj.stream_, (uint8_t *)devObj.dev_ptrs_[0], (uint8_t *)devObj.dev_ptrs_[1]);
    return 0;
}
