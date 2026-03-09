/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sk_common.cpp
 * \brief Utility functions for ELF symbol extraction and caching
 */


#include "sk_common.h"

#include <unordered_map>
#include <string>
#include <cstring>
#include <elf.h>
#include "sk_log.h"

namespace {
struct FuncSymbolInfo {
    std::string name;
    uint64_t size;
};
using SkFuncSymbolTable = std::unordered_map<uint64_t, FuncSymbolInfo>;
using SkBinSymbolTable = std::unordered_map<const char*, SkFuncSymbolTable>;


SkFuncSymbolTable BuildFuncSymbolTable(const char* binAddr, size_t binSize)
{
    SkFuncSymbolTable funcSymTable;
    if (binAddr == nullptr || binSize == 0) {
        SK_LOGE("Invalid bin parameters: binAddr=%p, binSize=%zu", binAddr, binSize);
        return funcSymTable;
    }
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(binAddr);
    uint64_t shCnt = ehdr->e_shnum;
    const Elf64_Shdr* shHdr = reinterpret_cast<const Elf64_Shdr*>(binAddr + ehdr->e_shoff);
    const char* shStrTbl = binAddr + shHdr[ehdr->e_shstrndx].sh_offset;

    const Elf64_Sym* symTbl = nullptr;
    size_t symSize = 0;
    const char* strTbl = nullptr;

    for (uint64_t i = 0; i < shCnt; ++i) {
        if (shHdr[i].sh_type == SHT_NULL || shHdr[i].sh_type == SHT_NOBITS) {
            continue;
        }
        const char* secName = shStrTbl + shHdr[i].sh_name;

        if (strcmp(".symtab", secName) == 0) {
            symTbl = reinterpret_cast<const Elf64_Sym*>(binAddr + shHdr[i].sh_offset);
            symSize = shHdr[i].sh_size;
        } else if (strcmp(".strtab", secName) == 0) {
            strTbl = binAddr + shHdr[i].sh_offset;
        }
    }

    if (symTbl == nullptr || strTbl == nullptr) {
        SK_LOGE("Failed to find symtab or strtab sections");
        return funcSymTable;
    }
    // build table
    size_t symCount = symSize / sizeof(Elf64_Sym);
    for (size_t i = 0; i < symCount; ++i) {
        if ((symTbl[i].st_info & 0xf) != STT_FUNC) { // filter non-function symbols
            continue;
        }
        if (symTbl[i].st_size != 0) { // filter out zero-size symbols
            const char* name = strTbl + symTbl[i].st_name;
            if (name != nullptr && name[0] != '\0') {
                funcSymTable[symTbl[i].st_value] = {name, symTbl[i].st_size};
            }
        }
    }
    SK_LOGI("BuildFuncSymbolTable: total %zu function symbols found", funcSymTable.size());
    return funcSymTable;
}

} // namespace

bool GetFuncSymbolInfo(const char* binAddr, size_t binSize, uint64_t funcAddr,
                       std::string& symbolName, uint64_t& funcSize)
{
    if (binAddr == nullptr || binSize == 0) {
        SK_LOGE("Invalid bin parameters: binAddr=%p, binSize=%zu", binAddr, binSize);
        return false;
    }
    static SkBinSymbolTable symbolTable;
    auto cacheIt = symbolTable.find(binAddr);
    if (cacheIt == symbolTable.end()) {
        SK_LOGI("GetFuncSymbolInfo: Building symbol table for binAddr=%p", binAddr);
        symbolTable[binAddr] = BuildFuncSymbolTable(binAddr, binSize);
        cacheIt = symbolTable.find(binAddr);
    }

    const auto& funcSymTable = cacheIt->second;
    auto it = funcSymTable.find(funcAddr);
    if (it != funcSymTable.end()) {
        symbolName = it->second.name;
        funcSize = it->second.size;
        SK_LOGI("GetFuncSymbolInfo: Found symbol: name=%s, addr=0x%lx, size=0x%lx",
                symbolName.c_str(), funcAddr, funcSize);
        return true;
    }
    SK_LOGW("GetFuncSymbolInfo: Function symbol not found for addr=0x%lx", funcAddr);
    return false;
}
