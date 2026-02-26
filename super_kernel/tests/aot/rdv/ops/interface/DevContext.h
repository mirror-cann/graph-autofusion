/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include "acl/acl.h"

#include "rms_norm_tiling.h"
#include "grouped_matmul_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling.h"
#include "dequant_swiglu_quant_tiling.h"
#include "dynamic_quant_tiling.h"

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

class DevContext {
public:
    DevContext(uint32_t device);
    ~DevContext();

    template<typename T, bool auto_add = true>
    T* set_ptr_(uint64_t size, T value, const char* mode = "copy") {
        // const uint64_t MAX_SIZE = 64 * 1024 * 1024;  // 64MB
        // if (size == 0 || size > MAX_SIZE) {
        //     std::cerr << "invalid size: " << size << std::endl;
        //     return nullptr;
        // }
        T* ptr = nullptr;
        CHECK_ACL(aclrtMalloc((void **)&ptr, size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST));
        if (strcmp(mode, "set") == 0) {
            CHECK_ACL(aclrtMemset(ptr, size * sizeof(T), value, size * sizeof(T)));
        } else if (strcmp(mode, "copy") == 0) {
            T* buf = (T*)malloc(size * sizeof(T));
            if (!buf) {
                std::cerr << "malloc failed for size " << size * sizeof(T) << std::endl;
                CHECK_ACL(aclrtFree(ptr));
                return nullptr;
            }
            for (uint64_t i = 0; i < size; i++) {
                buf[i] = value;
            }
            CHECK_ACL(aclrtMemcpy(ptr, size * sizeof(T), buf, size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
            free(buf);
        } else {
            std::cerr << "unsupported mode " << mode << std::endl;
        }
        if (auto_add) {
            dev_ptrs_.emplace_back(ptr);
        }
        return ptr;
    }

    std::vector<void *> dev_ptrs_;
    uint32_t dev_id_;
    aclrtStream stream_;
    void* workspace_;

public:
    void* x_in__;
    GMMTilingData op1_tiling;
    DequantSwigluQuantBaseTilingData op2_tiling;
    GMMTilingData op3_tiling;
    DynamicQuantTilingData op4_tiling;
};