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
    DynamicQuantTilingData tiling = {40, 7168, 16, 231, 230, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,{0, 0, 0, 0}
};
    int startId = devObj.dev_ptrs_.size();
    uint64_t x_shape[2] = {9216, tiling.rowLen};
    void* x_in__ = devObj.set_ptr_<int8_t>(x_shape[0] * x_shape[1] * 2, 1);
    void* scale_out_ = devObj.set_ptr_<float>(x_shape[0] , 0xff);
    void* y_out_ = devObj.set_ptr_<int8_t>(x_shape[0] * x_shape[1], 15);

    DynamicQuantKernel(
        tiling.coreNum, devObj.stream_,
        (uint8_t *)devObj.dev_ptrs_[startId + 0], nullptr, nullptr,
        (uint8_t *)devObj.dev_ptrs_[startId + 2], (uint8_t *)devObj.dev_ptrs_[startId + 1],
        (uint8_t *)devObj.workspace_, tiling
    );

    return 0;
}