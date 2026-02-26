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
    //     {8, 20, 0, 24, 256, 6144, 120576, 1, 1, 1, 0, 0, 1, 1, 9216, 0, 0, 0, 2048, 7168, 24, 0, 0, 0, 0, {0, 0, 0, 0}, },
    //     {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {7168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
    //     {20, 9216, 7168, 2048, 2048, 9216, 256, 2048, 128, 256, 128, 8, 8, 1, 1, 0, 0, 0, 0, 98304, 131072, 0, 1, 1, 1, 1, 4, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    // };
    GMMTilingData tiling = {
        {1, 20, 0, 24, 256, 6144, 120576, 1, 1, 1, 0, 0, 1, 1, 9216, 0, 0, 0, 2048, 7168, 24, 0, 0, 0, 0, {0, 0, 0, 0}, },
        {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {7168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
        {20, 9216, 7168, 2048, 2048, 9216, 256, 2048, 128, 256, 128, 8, 8, 1, 1, 0, 0, 0, 0, 98304, 131072, 0, 1, 1, 1, 1, 4, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    };
    
    int64_t x_shape[2] = {(int64_t)tiling.mmTilingData.M, (int64_t)tiling.mmTilingData.Ka};
    devObj.set_ptr_<uint8_t>(x_shape[0] * x_shape[1], 1);
    void* x_dev[1] = {devObj.dev_ptrs_.back()};
    void* x_in__ = create_tensor_list(2, x_shape, 1, x_dev);

    int startId = devObj.dev_ptrs_.size() - 1;

    int64_t weight_shape[3] = {tiling.gmmBaseParams.groupNum, (int64_t)tiling.mmTilingData.Kb, (int64_t)tiling.mmTilingData.N};
    devObj.set_ptr_<uint8_t>(weight_shape[0] * weight_shape[1] * weight_shape[2], 2);
    void* weight_dev[1] = {devObj.dev_ptrs_.back()};
    void* weight_in__ = create_tensor_list(3, weight_shape, 1, weight_dev);

    int64_t scale_shape[2] = {(int64_t)tiling.gmmBaseParams.groupNum, (int64_t)tiling.mmTilingData.N};
    devObj.set_ptr_<uint8_t>(scale_shape[0] * scale_shape[1] * 2, 3);
    void* scale_dev[1] = {devObj.dev_ptrs_.back()};
    void* scale_in__ = create_tensor_list(2, scale_shape, 1, scale_dev);

    void* group_index_in__ = devObj.set_ptr_<uint64_t>((int64_t)tiling.gmmBaseParams.groupNum, tiling.mmTilingData.M / tiling.gmmBaseParams.groupNum, "copy");

    int64_t pre_token_scale_shape[1] = {(int64_t)tiling.mmTilingData.M};
    devObj.set_ptr_<uint8_t>(pre_token_scale_shape[0] * 4, 4);
    void* pre_token_scale_dev[1] = {devObj.dev_ptrs_.back()};
    void* pre_token_scale_in__ = create_tensor_list(1, pre_token_scale_shape, 1, pre_token_scale_dev);

    int64_t y_shape[2] = {(int64_t)tiling.mmTilingData.M, (int64_t)tiling.mmTilingData.N};
    devObj.set_ptr_<uint8_t>(y_shape[0] * y_shape[1] * 4, 15);
    void* y_dev[1] = {devObj.dev_ptrs_.back()};
    void* y_out_ = create_tensor_list(2, y_shape, 1, y_dev);

    GroupedMatmulKernelV3(tiling.mmTilingData.usedCoreNum, devObj.stream_,
                        (uint8_t *)x_in__, (uint8_t *)weight_in__, nullptr, (uint8_t *)scale_in__,
                         nullptr, nullptr, nullptr,
                        (uint8_t *)group_index_in__, (uint8_t *)pre_token_scale_in__, (uint8_t *)y_out_,
                        (uint8_t *)devObj.workspace_, tiling);
    return 0;
}
