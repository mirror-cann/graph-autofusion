/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_V35_RUNTIME_STUB_H
#define AUTOFUSE_V35_RUNTIME_STUB_H

// Include the base RuntimeStub using angle brackets to avoid searching the current directory.
// tests/depends/runtime/src is in the include path, so <runtime_stub.h> resolves correctly.
#include <runtime_stub.h>

namespace af{
class RuntimeStubV2 : public ge::RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    (void)strcpy_s(version, maxLen, "Ascend910_9591");
    return RT_ERROR_NONE;
  }

  rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) override {
    (void)label;
    (void)key;
    (void)strcpy_s(val, maxLen, "3510");
    return RT_ERROR_NONE;
  }
};
}  // namespace af

#endif  // AUTOFUSE_V35_RUNTIME_STUB_H
