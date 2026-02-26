/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "tensor_list.h"
#include <iostream>
#include <stdlib.h>
#include "acl/acl.h"

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

void *create_tensor_list(int dim, int64_t shape[], int cnt, void *dev[])
{
  // 头部记录tensor list描述信息的总大小
  size_t desc_size = sizeof(int64_t) + cnt * (((dim + 1) * sizeof(int64_t)) + sizeof(void*));
  void *host_ptr = malloc(desc_size);
  *(int64_t *)host_ptr = sizeof(int64_t) + cnt * ((dim + 1) * sizeof(int64_t));
  printf("tensor list ptr offset %ld:\n", *(int64_t *)host_ptr);
  for (int i = 0; i < cnt; i++) {
    // 首空间记录每个tensor的维度信息
    int64_t *cur_dim = (int64_t *)((char *)host_ptr + sizeof(int64_t) + i * ((dim + 1) * sizeof(int64_t)));
    *cur_dim = dim;
    cur_dim = cur_dim + 1;
    for (int j = 0; j < dim; j++) {
      cur_dim[j] = shape[j];
    }
    // 尾空间记录每个tensor的data数据地址
    void **ptrs = (void **)((char *)host_ptr + sizeof(int64_t) + cnt * ((dim + 1) * sizeof(int64_t)) + i * sizeof(void*));
    ptrs[i] = dev[i];
    printf("[%d] tensor ptr %p\n", i, dev[i]);
  }
  void *dev_ptr;
  CHECK_ACL(aclrtMalloc((void **)&dev_ptr, desc_size, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(dev_ptr, desc_size, host_ptr, desc_size, ACL_MEMCPY_HOST_TO_DEVICE));
  free(host_ptr);
  return dev_ptr;
}
