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
    // DevContext devObj(0);
    // int startId = devObj.dev_ptrs_.size();
    // devObj.dev_ptrs_.resize(startId+9);
    // uint16_t buf[1024];
    // for (int i = 0; i < 1024; i++) {
    //     buf[i] = 0x3c00;
    // }
    // for (auto i = 0; i < 9; i++) {
    //     CHECK_ACL(aclrtMalloc((void **)&(devObj.dev_ptrs_[startId+i]), 4096 * 4096, ACL_MEM_MALLOC_HUGE_FIRST));
    //     CHECK_ACL(aclrtMemcpy(devObj.dev_ptrs_[startId+i], 2048, buf, 2048, ACL_MEMCPY_HOST_TO_DEVICE));
    // }
    // SwiGluTilingData tiling = {1,0,32,16,1,16,0,0,1,0,0,0,0,32,{0,0,0,0}};
    // DequantSwiGluQuantDynamicKernel(32, devObj.stream_,
    //                                      (uint8_t *)devObj.dev_ptrs_[startId+0], (uint8_t *)devObj.dev_ptrs_[startId+1],
    //                                      (uint8_t *)devObj.dev_ptrs_[startId+2], (uint8_t *)devObj.dev_ptrs_[startId+3],
    //                                      (uint8_t *)devObj.dev_ptrs_[startId+4], (uint8_t *)devObj.dev_ptrs_[startId+5],
    //                                      (uint8_t *)devObj.dev_ptrs_[startId+6], (uint8_t *)devObj.dev_ptrs_[startId+8],
    //                                      (uint8_t *)devObj.dev_ptrs_[startId+7],(uint8_t *)devObj.workspace_, tiling);

    DevContext devObj(0);
    // DequantSwigluQuantBaseTilingData tiling = {9216, 4096, 2048, 4, 2048, 36, 36, 8, 0, 1, 1, 0, 1, 1, 0,0,0,0,0,7.0,1.7020000219345093, 1.0,{0,0,0,0}};
    DequantSwigluQuantBaseTilingData tiling = {9216, 4096, 2048, 4, 2048, 36, 36, 1, 0, 1, 1, 0, 1, 1, 0,0,0,0,0,7.0,1.7020000219345093, 1.0,{0,0,0,0}};
    void* x_in__ = devObj.set_ptr_<uint32_t>(tiling.inDimx * tiling.inDimy, 100);

    int startId = devObj.dev_ptrs_.size() - 1;
    void* weight_scale_in__ = devObj.set_ptr_<float>(tiling.inGroupNum * tiling.inDimy, 2);
    void* activation_scale_in__ = devObj.set_ptr_<float>(tiling.inDimx, 2);
    void* quant_scale_in__ = devObj.set_ptr_<float>(tiling.inGroupNum * tiling.outDimy, 2);
    // void* group_index_in__ = devObj.set_ptr_<uint64_t>(tiling.inGroupNum, tiling.inDimx / tiling.inGroupNum, "copy");
    void* group_index_in__ = devObj.set_ptr_<uint64_t>(tiling.inGroupNum, 1, "copy");
    void* scale_out__ = devObj.set_ptr_<float>(tiling.inDimx, 0);
    void* y_out__ = devObj.set_ptr_<uint8_t>(tiling.inDimx * tiling.outDimy, 15);

    DequantSwiGluQuantDynamicKernel(tiling.usedCoreNum, devObj.stream_,
                                        (uint8_t *)devObj.dev_ptrs_[startId + 0], (uint8_t *)devObj.dev_ptrs_[startId + 1],
                                        (uint8_t *)devObj.dev_ptrs_[startId + 2], nullptr,
                                        (uint8_t *)devObj.dev_ptrs_[startId + 3], nullptr,
                                        (uint8_t *)devObj.dev_ptrs_[startId + 4], (uint8_t *)devObj.dev_ptrs_[startId + 6],
                                        (uint8_t *)devObj.dev_ptrs_[startId + 5], (uint8_t *)devObj.workspace_, tiling);
    return 0;
}