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
#include <set>
#include <unordered_set>

DevContext::DevContext(uint32_t device) {
    dev_id_ = device;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device));
    printf("device context created!\n");
    CHECK_ACL(aclrtCreateStream(&stream_));
    CHECK_ACL(aclrtSetOpExecuteTimeOut(1));
    workspace_ = set_ptr_<uint8_t>(64 * 1024 * 1024, 0xaa);
}

DevContext::~DevContext() {
    printf("sync kernel\n");
    CHECK_ACL(aclrtSynchronizeStreamWithTimeout(stream_, 1000));
    CHECK_ACL(aclrtDestroyStream(stream_));
    uint16_t buf[32];
    printf("dev_ptrs_size=%lu\n", dev_ptrs_.size());
    CHECK_ACL(aclrtMemcpy(buf, sizeof(buf), dev_ptrs_[dev_ptrs_.size() - 1], sizeof(buf), ACL_MEMCPY_DEVICE_TO_HOST));
    printf("Result:\n");
    for (int i = 0; i < 32; i++) {
        printf("%04x ", buf[i]);
    }
    printf("\n");
    std::unordered_set<void*> unique_ptrs;
    for (int i = 0; i < dev_ptrs_.size(); i++){
        // printf("free dev_ptrs_[%d]=%p\n", i, dev_ptrs_[i]);
        if(unique_ptrs.count(dev_ptrs_[i])){
            // printf("id:%d, ptr:%p already freed, skip\n", i, dev_ptrs_[i]);
            continue;
        }
        CHECK_ACL(aclrtFree(dev_ptrs_[i]));
        unique_ptrs.insert(dev_ptrs_[i]);
    }
    dev_ptrs_.clear();
        
    CHECK_ACL(aclrtResetDevice(dev_id_));
    CHECK_ACL(aclFinalize());
    printf("device context destroy!\n");
}