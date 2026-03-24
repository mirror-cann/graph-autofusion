/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Source Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SK_FILE_GUARD_H
#define SK_FILE_GUARD_H

#include <cstdio>

// RAII 封装 FILE*，确保异常安全
class FileGuard {
public:
    explicit FileGuard(const char* filename = nullptr, const char* mode = "rb") {
        if (filename != nullptr) {
            filePtr = fopen(filename, mode);
        } else {
            filePtr = nullptr;
        }
    }
    ~FileGuard() { Close(); }
    
    FileGuard(const FileGuard&) = delete;
    FileGuard& operator=(const FileGuard&) = delete;
    
    FileGuard(FileGuard&& other) noexcept : filePtr(other.filePtr) {
        other.filePtr = nullptr;
    }
    
    FILE* Get() const { return filePtr; }
    bool IsValid() const { return filePtr != nullptr; }
    operator bool() const { return IsValid(); }
    
    // 打开文件
    bool Open(const char* filename, const char* mode) {
        Close();
        filePtr = fopen(filename, mode);
        return IsValid();
    }
    
    void Close() {
        if (filePtr != nullptr) {
            fclose(filePtr);
            filePtr = nullptr;
        }
    }
    
    // 释放文件句柄的所有权，调用者负责关闭
    FILE* Release() {
        FILE* fp = filePtr;
        filePtr = nullptr;
        return fp;
    }
    
private:
    FILE* filePtr;
};

// 从 MsprofGetPath 返回的路径中提取profiling路径, 
// 返回路径类似 父目录/profiling/0002_3675077_20260321093945052_ascend_pt/PROF_000001_20260321093945081_03675077MGDRRBBN
static std::string GetBasePath() {
    const char* pathRaw = MsprofGetPath();
    if (pathRaw == nullptr || pathRaw[0] == '\0') {
        SK_LOGE("[sk time profiling] MsprofGetPath returned empty path\n");
        SK_LOGI("[sk time profiling] Profiler should start before than net start, Please check it\n");
        return "";
    }
    std::string path(pathRaw);

    const std::string suffix = "/mindstudio_profiler_output";
    if (path.size() >= suffix.size() &&
        path.substr(path.size() - suffix.size()) == suffix) {
        path = path.substr(0, path.size() - suffix.size());
    }

    SK_LOGI("[sk time profiling] Output directory: %s\n", path.c_str());
    return path;
}

#endif // SK_FILE_GUARD_H
