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

std::pair<int, int> op1_memory_malloc(DevContext &devObj, const std::vector<int>& input_shape, uint32_t group_num = 8, uint64_t group_value = 1) {
    int startId = devObj.dev_ptrs_.size();

    // devObj.op1_tiling = {
    //     {group_num, 20, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 9216, 0, 0, 0, 7168, 4096, 0, 0, 0, 0, 0, {0, 0, 0, 0}, },
    //     {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {7168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
    //     {20, 9216, 4096, 7168, 7168, 9216, 256, 7168, 128, 256, 128, 8, 8, 1, 1, 0, 0, 0, 0, 98304, 131072, 0, 1, 1, 1, 1, 4, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    // };
    devObj.op1_tiling = {
        {group_num, 24, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 9216, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, {0, 0, 0, 0}, },
        {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {7168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
        {24, 9216, 4096, 7168, 7168, 9216, 256, 7168, 128, 256, 128, 8, 8, 1, 1, 0, 0, 0, 0, 98304, 131072, 0, 1, 1, 1, 1, 4, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    };

    int64_t x_shape[2] = {(int64_t)devObj.op1_tiling.mmTilingData.M, (int64_t)devObj.op1_tiling.mmTilingData.Ka};
    void* x_ptr = devObj.dev_ptrs_.back();
    void* x_dev[1] = {x_ptr};
    void* x_in__ = create_tensor_list(2, x_shape, 1, x_dev);
    devObj.dev_ptrs_.emplace_back(x_in__);

    int64_t weight_shape[3] = {devObj.op1_tiling.gmmBaseParams.groupNum, (int64_t)devObj.op1_tiling.mmTilingData.Kb, (int64_t)devObj.op1_tiling.mmTilingData.N};
    void* weight_ptr = devObj.set_ptr_<int8_t, false>(weight_shape[0] * weight_shape[1] * weight_shape[2], 2);
    void* weight_dev[1] = {weight_ptr};
    void* weight_in__ = create_tensor_list(3, weight_shape, 1, weight_dev);
    devObj.dev_ptrs_.emplace_back(weight_in__);

    void* group_index_in__ = devObj.set_ptr_<uint64_t, false>(devObj.op1_tiling.gmmBaseParams.groupNum, group_value, "copy");
    devObj.dev_ptrs_.emplace_back(group_index_in__);

    int64_t y_shape[2] = {(int64_t)devObj.op1_tiling.mmTilingData.M, (int64_t)devObj.op1_tiling.mmTilingData.N};
    void* y_ptr = devObj.set_ptr_<int32_t, false>(y_shape[0] * y_shape[1], 15);
    void* y_dev[1] = {y_ptr};
    void* y_out__ = create_tensor_list(2, y_shape, 1, y_dev);
    devObj.dev_ptrs_.emplace_back(y_out__);
    devObj.dev_ptrs_.emplace_back(y_ptr);
    return {startId, devObj.dev_ptrs_.size() - startId};
}

std::pair<int, int> op2_memory_malloc(DevContext &devObj, const std::vector<int>& input_shape, int64_t group_num = 8, uint64_t group_value = 1){
    int startId = devObj.dev_ptrs_.size();
    
    devObj.op2_tiling = {9216, 4096, 2048, 4, 2048, 36, 36, group_num, 0, 1, 1, 0, 1, 1, 0,0,0,0,0,7.0,1.7020000219345093, 1.0,{0,0,0,0}};
    void* x_in__ = devObj.dev_ptrs_.back();
    devObj.dev_ptrs_.emplace_back(x_in__);
    void* weight_scale_in__ = devObj.set_ptr_<float>(devObj.op2_tiling.inGroupNum * devObj.op2_tiling.inDimy, 2);
    void* activation_scale_in__ = devObj.set_ptr_<float>(devObj.op2_tiling.inDimx, 2);
    void* quant_scale_in__ = devObj.set_ptr_<float>(devObj.op2_tiling.inGroupNum * devObj.op2_tiling.outDimy, 2);
    void* group_index_in__ = devObj.set_ptr_<uint64_t>(devObj.op2_tiling.inGroupNum, group_value, "copy");
    void* scale_out__ = devObj.set_ptr_<float>(devObj.op2_tiling.inDimx, 0);
    void* y_out__ = devObj.set_ptr_<uint8_t>(devObj.op2_tiling.inDimx * devObj.op2_tiling.outDimy, 0);
    return {startId, devObj.dev_ptrs_.size() - startId};
}

std::pair<int, int> op3_memory_malloc(DevContext &devObj, const std::vector<int>& input_shape, uint32_t group_num = 8, uint64_t group_value = 1){
    int startId = devObj.dev_ptrs_.size();

    // devObj.op3_tiling = {
    //     {group_num, 20, 0, 24, 256, 6144, 120576, 1, 1, 1, 0, 0, 1, 1, 9216, 0, 0, 0, 2048, 7168, 24, 0, 0, 0, 0, {0, 0, 0, 0}, },
    //     {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {7168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
    //     {20, 9216, 7168, 2048, 2048, 9216, 256, 2048, 128, 256, 128, 8, 8, 1, 1, 0, 0, 0, 0, 98304, 131072, 0, 1, 1, 1, 1, 4, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    // };
    devObj.op3_tiling = {
        {group_num, 24, 0, 24, 256, 6144, 120576, 1, 1, 1, 0, 0, 1, 1, 9216, 0, 0, 0, 0, 0, 24, 0, 0, 1, 0, {0, 0, 0, 0}, },
        {{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {7168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, },
        {24, 9216, 7168, 2048, 2048, 9216, 256, 2048, 128, 256, 128, 8, 8, 1, 1, 0, 0, 0, 0, 98304, 131072, 0, 1, 1, 1, 1, 4, 4, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }
    };

    int64_t x_shape[2] = {(int64_t)devObj.op3_tiling.mmTilingData.M, (int64_t)devObj.op3_tiling.mmTilingData.Ka};
    void* x_ptr = devObj.dev_ptrs_.back();
    void* x_dev[1] = {x_ptr};
    void* x_in__ = create_tensor_list(2, x_shape, 1, x_dev);
    devObj.dev_ptrs_.emplace_back(x_in__);

    int64_t weight_shape[3] = {devObj.op3_tiling.gmmBaseParams.groupNum, (int64_t)devObj.op3_tiling.mmTilingData.Kb, (int64_t)devObj.op3_tiling.mmTilingData.N};
    void* weight_ptr = devObj.set_ptr_<uint8_t, false>(weight_shape[0] * weight_shape[1] * weight_shape[2], 2);
    void* weight_dev[1] = {weight_ptr};
    void* weight_in__ = create_tensor_list(3, weight_shape, 1, weight_dev);
    devObj.dev_ptrs_.emplace_back(weight_in__);

    int64_t scale_shape[2] = {(int64_t)devObj.op3_tiling.gmmBaseParams.groupNum, (int64_t)devObj.op3_tiling.mmTilingData.N};
    void* scale_ptr = devObj.set_ptr_<uint8_t, false>(scale_shape[0] * scale_shape[1] * 2, 3);
    void* scale_dev[1] = {scale_ptr};
    void* scale_in__ = create_tensor_list(2, scale_shape, 1, scale_dev);
    devObj.dev_ptrs_.emplace_back(scale_in__);

    void* group_index_in__ = devObj.set_ptr_<uint64_t, false>((int64_t)devObj.op3_tiling.gmmBaseParams.groupNum, group_value, "copy");
    devObj.dev_ptrs_.emplace_back(group_index_in__);

    int64_t pre_token_scale_shape[1] = {(int64_t)devObj.op3_tiling.mmTilingData.M};
    void* pre_token_scale_ptr = devObj.set_ptr_<uint8_t, false>(pre_token_scale_shape[0] * 4, 4);
    void* pre_token_scale_dev[1] = {pre_token_scale_ptr};
    void* pre_token_scale_in__ = create_tensor_list(1, pre_token_scale_shape, 1, pre_token_scale_dev);
    devObj.dev_ptrs_.emplace_back(pre_token_scale_in__);

    int64_t y_shape[2] = {(int64_t)devObj.op3_tiling.mmTilingData.M, (int64_t)devObj.op3_tiling.mmTilingData.N};
    void* y_ptr = devObj.set_ptr_<uint8_t, false>(y_shape[0] * y_shape[1] * 4, 15);
    void* y_dev[1] = {y_ptr};
    void* y_out_ = create_tensor_list(2, y_shape, 1, y_dev);
    devObj.dev_ptrs_.emplace_back(y_out_);
    devObj.dev_ptrs_.emplace_back(y_ptr);
    return {startId, devObj.dev_ptrs_.size() - startId};
}

std::pair<int, int> op4_memory_malloc(DevContext &devObj, const std::vector<int>& input_shape, uint32_t group_num = 8, uint64_t group_value = 1){
    int startId = devObj.dev_ptrs_.size();
    // devObj.op4_tiling = {40, 7168, 16, 231, 230, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,{0, 0, 0, 0}};
    devObj.op4_tiling = {48, 7168, 0, 192, 192, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,{0, 0, 0, 0}};
    void* x_in__ = devObj.dev_ptrs_.back();
    devObj.dev_ptrs_.emplace_back(x_in__);
    void* scale_out_ = devObj.set_ptr_<float>(input_shape[0] , 0x10);
    void* y_out_ = devObj.x_in__;
    devObj.dev_ptrs_.emplace_back(y_out_);
    return {startId, devObj.dev_ptrs_.size() - startId};
}

std::vector<std::pair<int,int>> network__memory_malloc(DevContext &devObj){
    // gen input tensor list
    std::vector<int> input_shape = {9216, 7168};
    uint64_t input_size = 1;
    for (auto dim : input_shape) {
        input_size *= dim;
    }
    devObj.x_in__ = devObj.set_ptr_<int8_t>(input_size, 1);
    printf("input tensor created! dev_ptr size: %ld\n", devObj.dev_ptrs_.size());
    // uint32_t group_num = 8;
    // uint64_t group_value = input_shape[0] / group_num;
    uint32_t group_num = 1;
    uint64_t group_value = 1;
    
    std::vector<std::pair<int,int>> op_params_idx_size;
    // gen op 1 params
    auto op1_idx_size = op1_memory_malloc(devObj, input_shape, group_num, group_value);
    op_params_idx_size.emplace_back(op1_idx_size);
    printf("op1 memory malloc done! dev_ptr size: %ld\n", devObj.dev_ptrs_.size());
    // assert(op1_idx_size.second == 5 && "dev_ptrs_ size error!");
    // gen op 2 params
    auto op2_idx_size = op2_memory_malloc(devObj, input_shape, group_num, group_value);
    op_params_idx_size.emplace_back(op2_idx_size);
    printf("op2 memory malloc done! dev_ptr size: %ld\n", devObj.dev_ptrs_.size());
    // assert(op2_idx_size.second == 7 && "dev_ptrs_ size error!");
    // gen op 3 params
    auto op3_idx_size = op3_memory_malloc(devObj, input_shape, group_num, group_value);
    op_params_idx_size.emplace_back(op3_idx_size);
    printf("op3 memory malloc done! dev_ptr size: %ld\n", devObj.dev_ptrs_.size());
    // assert(op3_idx_size.second == 7 && "dev_ptrs_ size error!");
    // gen op 4 params
    auto op4_idx_size = op4_memory_malloc(devObj, input_shape, group_num, group_value);
    op_params_idx_size.emplace_back(op4_idx_size);
    printf("op4 memory malloc done! dev_ptr size: %ld\n", devObj.dev_ptrs_.size());
    // assert(op4_idx_size.second == 3 && "dev_ptrs_ size error!");
    return op_params_idx_size;
}

void launch_network(DevContext &devObj, const std::vector<std::pair<int,int>>& op_params_idx_size, int repeat_cnt = 1){
    for (int i = 0; i < repeat_cnt; i++) {
        printf("network launch iter %d\n", i);
        // op 1
        {
            auto& tiling = devObj.op1_tiling;
            auto startId = op_params_idx_size[0].first;
            GroupedMatmulKernelV2(tiling.gmmBaseParams.coreNum, devObj.stream_,
                            (uint8_t *)devObj.dev_ptrs_[startId + 0], (uint8_t *)devObj.dev_ptrs_[startId + 1], nullptr, nullptr,
                            nullptr, nullptr, nullptr,
                            (uint8_t *)devObj.dev_ptrs_[startId + 2], nullptr, (uint8_t *)devObj.dev_ptrs_[startId + 3],
                            (uint8_t *)devObj.workspace_, tiling);
        }
        // op 2
        {
            auto& tiling = devObj.op2_tiling;
            auto startId = op_params_idx_size[1].first;
            DequantSwiGluQuantDynamicKernel(tiling.usedCoreNum, devObj.stream_,
                                            (uint8_t *)devObj.dev_ptrs_[startId + 0], (uint8_t *)devObj.dev_ptrs_[startId + 1],
                                            (uint8_t *)devObj.dev_ptrs_[startId + 2], nullptr,
                                            (uint8_t *)devObj.dev_ptrs_[startId + 3], nullptr,
                                            (uint8_t *)devObj.dev_ptrs_[startId + 4], (uint8_t *)devObj.dev_ptrs_[startId + 6],
                                            (uint8_t *)devObj.dev_ptrs_[startId + 5], (uint8_t *)devObj.workspace_, tiling);
        }
        // op 3
        {
            auto& tiling = devObj.op3_tiling;
            auto startId = op_params_idx_size[2].first;
            GroupedMatmulKernelV3(tiling.gmmBaseParams.coreNum, devObj.stream_,
                            (uint8_t *)devObj.dev_ptrs_[startId + 0], (uint8_t *)devObj.dev_ptrs_[startId + 1], nullptr, (uint8_t *)devObj.dev_ptrs_[startId + 2],
                            nullptr, nullptr, nullptr,
                            (uint8_t *)devObj.dev_ptrs_[startId + 3], (uint8_t *)devObj.dev_ptrs_[startId + 4], (uint8_t *)devObj.dev_ptrs_[startId + 5],
                            (uint8_t *)devObj.workspace_, tiling);
        }
        // op 4
        {
            auto& tiling = devObj.op4_tiling;
            auto startId = op_params_idx_size[3].first;
            DynamicQuantKernel(tiling.coreNum, devObj.stream_,
                (uint8_t *)devObj.dev_ptrs_[startId + 0], nullptr, nullptr,
                (uint8_t *)devObj.dev_ptrs_[startId + 2], (uint8_t *)devObj.dev_ptrs_[startId + 1],
                (uint8_t *)devObj.workspace_, tiling);
        }
        // op nop
        ClearOpsKernelLaunch(24, devObj.stream_);
    }
}

void gen_rms_kernel_func(DevContext &devObj){
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
    RmsNormKernel(1, devObj.stream_, (uint8_t *)devObj.dev_ptrs_[startId], (uint8_t *)devObj.dev_ptrs_[startId+1],
                  (uint8_t *)devObj.dev_ptrs_[startId+3], (uint8_t *)devObj.dev_ptrs_[startId+2], tiling);
}

void gen_grouped_matmul_func(DevContext &devObj){
    int startId = devObj.dev_ptrs_.size();
    devObj.dev_ptrs_.resize(startId + 8);
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
}

void gen_weight_quant_batch_matmul_v2_func(DevContext &devObj){
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
    WeightQuantBatchMatmulV2MsdTilingData tilingMc = {1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 32, 320, 32, 320, 256, 0, 0, {1, 64, 256, 320, 320, 64, 256, 320, 64, 256, 128, 3, 3, 1, 1, 0, 0, 0, 0, 122880, 65536, 0, 1, 1, 1, 1, 3, 3, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }};
    // todo: origin block dim 16, for compatibility test set 20.
    WeightQuantBatchMatmulV2Kernel(20, devObj.stream_, (uint8_t *)devObj.dev_ptrs_[startId], (uint8_t *)devObj.dev_ptrs_[startId+1], (uint8_t *)devObj.dev_ptrs_[startId+2], (uint8_t *)devObj.dev_ptrs_[startId+3], nullptr, nullptr, nullptr, (uint8_t *)devObj.dev_ptrs_[startId+4], (uint8_t *)devObj.workspace_, tilingMc);
}

void gen_matmul_add_func(DevContext &devObj){
    int startId = devObj.dev_ptrs_.size();
    devObj.dev_ptrs_.resize(startId + 2);
    uint8_t buf[1024];
    for (int i = 0; i < 1024; i++) {
        buf[i] = 1;
    }
    for (auto i = 0; i < 2; i++) {
        CHECK_ACL(aclrtMalloc((void **)&(devObj.dev_ptrs_[startId+i]), 4096 * 4096 * 2, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemset(devObj.dev_ptrs_[startId+i], 4096, 1, 4096));
    }
    MatmulAdd(20, devObj.stream_, (uint8_t *)devObj.dev_ptrs_[startId], (uint8_t *)devObj.dev_ptrs_[startId+1]);
}

void gen_dequant_swiglu_quant_func(DevContext &devObj){
    int startId = devObj.dev_ptrs_.size();
    devObj.dev_ptrs_.resize(startId + 9);
    uint16_t buf[1024];
    for (int i = 0; i < 1024; i++)
    {
        buf[i] = 0x3c00;
    }
    for (auto i = 0; i < 9; i++)
    {
        CHECK_ACL(aclrtMalloc((void **)&(devObj.dev_ptrs_[startId + i]), 4096 * 4096, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(devObj.dev_ptrs_[startId + i], 2048, buf, 2048, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    SwiGluTilingData tiling = {1, 0, 32, 16, 1, 16, 0, 0, 1, 0, 0, 0, 0, 32, {0, 0, 0, 0}};
    DequantSwiGluQuantDynamicKernel(32, devObj.stream_,
                                        (uint8_t *)devObj.dev_ptrs_[startId + 0], (uint8_t *)devObj.dev_ptrs_[startId + 1],
                                        (uint8_t *)devObj.dev_ptrs_[startId + 2], (uint8_t *)devObj.dev_ptrs_[startId + 3],
                                        (uint8_t *)devObj.dev_ptrs_[startId + 4], (uint8_t *)devObj.dev_ptrs_[startId + 5],
                                        (uint8_t *)devObj.dev_ptrs_[startId + 6], (uint8_t *)devObj.dev_ptrs_[startId + 8],
                                        (uint8_t *)devObj.dev_ptrs_[startId + 7], (uint8_t *)devObj.workspace_, tiling);
}