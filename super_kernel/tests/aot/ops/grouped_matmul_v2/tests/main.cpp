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
    // GMMTilingData tiling = {
    //     {8, 20, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 9216, 0, 0, 0, 7168, 4096, 0, 0, 0, 0, 0, {0, 0, 0, 0}, },
    //     {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {7168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
    //     {20, 9216, 4096, 7168, 7168, 9216, 256, 7168, 128, 256, 128, 8, 8, 1, 1, 0, 0, 0, 0, 98304, 131072, 0, 1, 1, 1, 1, 4, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    // };
    GMMTilingData tiling = {
        {1, 20, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 9216, 0, 0, 0, 7168, 4096, 0, 0, 0, 0, 0, {0, 0, 0, 0}, },
        {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {7168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
        {20, 9216, 4096, 7168, 7168, 9216, 256, 7168, 128, 256, 128, 8, 8, 1, 1, 0, 0, 0, 0, 98304, 131072, 0, 1, 1, 1, 1, 4, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    };
    
    int startId = devObj.dev_ptrs_.size();

    int64_t x_shape[2] = {(int64_t)tiling.mmTilingData.M, (int64_t)tiling.mmTilingData.Ka};
    void* x_ptr = devObj.set_ptr_<int8_t, false>(x_shape[0] * x_shape[1], 1);
    void* x_dev[1] = {x_ptr};
    void* x_in__ = create_tensor_list(2, x_shape, 1, x_dev);
    devObj.dev_ptrs_.emplace_back(x_in__);

    int64_t weight_shape[3] = {tiling.gmmBaseParams.groupNum, (int64_t)tiling.mmTilingData.Kb, (int64_t)tiling.mmTilingData.N};
    void* weight_ptr = devObj.set_ptr_<int8_t, false>(weight_shape[0] * weight_shape[1] * weight_shape[2], 2);
    void* weight_dev[1] = {weight_ptr};
    void* weight_in__ = create_tensor_list(3, weight_shape, 1, weight_dev);
    devObj.dev_ptrs_.emplace_back(weight_in__);

    // void* group_index_in__ = devObj.set_ptr_<uint64_t, false>(tiling.gmmBaseParams.groupNum, tiling.mmTilingData.M / tiling.gmmBaseParams.groupNum, "copy");
    void* group_index_in__ = devObj.set_ptr_<uint64_t, false>(tiling.gmmBaseParams.groupNum, 1, "copy");
    devObj.dev_ptrs_.emplace_back(group_index_in__);

    int64_t y_shape[2] = {(int64_t)tiling.mmTilingData.M, (int64_t)tiling.mmTilingData.N};
    void* y_ptr = devObj.set_ptr_<int32_t, false>(y_shape[0] * y_shape[1], 15);
    void* y_dev[1] = {y_ptr};
    void* y_out__ = create_tensor_list(2, y_shape, 1, y_dev);
    devObj.dev_ptrs_.emplace_back(y_out__);
    devObj.dev_ptrs_.emplace_back(y_ptr);

    GroupedMatmulKernelV2(tiling.mmTilingData.usedCoreNum, devObj.stream_,
                        (uint8_t *)devObj.dev_ptrs_[startId + 0], (uint8_t *)devObj.dev_ptrs_[startId + 1], nullptr, nullptr,
                        nullptr, nullptr, nullptr,
                        (uint8_t *)devObj.dev_ptrs_[startId + 2], nullptr, (uint8_t *)devObj.dev_ptrs_[startId + 3],
                        (uint8_t *)devObj.workspace_, tiling);
    return 0;
}
