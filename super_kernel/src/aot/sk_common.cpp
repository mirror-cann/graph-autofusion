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

#include <acl/acl.h>
#include <unordered_map>
#include <string>
#include <cstring>
#include <elf.h>
#include "sk_log.h"

bool IsDav3510Soc()
{
    const char* socName = aclrtGetSocName();
    return socName != nullptr && strstr(socName, "Ascend950") != nullptr;
}

namespace {

enum class SymBindType : uint8_t {
    LOCAL = 0,
    GLOBAL = 1,
    WEAK = 2,
};

const char* SymBindTypeToStr(SymBindType bindType) {
    switch (bindType) {
        case SymBindType::LOCAL:  
            return "LOCAL";
        case SymBindType::GLOBAL: 
            return "GLOBAL";
        case SymBindType::WEAK:    
            return "WEAK";
        default:                   
            return "UNKNOWN";
    }
}

struct FuncSymbolInfo {
    std::string name;
    uint64_t size;
    SymBindType bindType;
};
using SkFuncSymbolTable = std::unordered_map<uint64_t, FuncSymbolInfo>;
using SkBinSymbolTable = std::unordered_map<const char*, SkFuncSymbolTable>;

struct ElfSymbolTables {
    const Elf64_Sym* symTbl;
    size_t symSize;
    const char* strTbl;
};

/**
 * @brief Find symbol tables in ELF binary
 */
static bool FindElfSymbolTables(const char* binAddr, ElfSymbolTables& tables) {
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(binAddr);
    const Elf64_Shdr* shHdr = reinterpret_cast<const Elf64_Shdr*>(binAddr + ehdr->e_shoff);
    const char* shStrTbl = binAddr + shHdr[ehdr->e_shstrndx].sh_offset;

    tables.symTbl = nullptr;
    tables.symSize = 0;
    tables.strTbl = nullptr;

    for (uint64_t i = 0; i < ehdr->e_shnum; ++i) {
        if (shHdr[i].sh_type == SHT_NULL || shHdr[i].sh_type == SHT_NOBITS) {
            continue;
        }
        const char* secName = shStrTbl + shHdr[i].sh_name;
        if (strcmp(".symtab", secName) == 0) {
            tables.symTbl = reinterpret_cast<const Elf64_Sym*>(binAddr + shHdr[i].sh_offset);
            tables.symSize = shHdr[i].sh_size;
        } else if (strcmp(".strtab", secName) == 0) {
            tables.strTbl = binAddr + shHdr[i].sh_offset;
        }
    }
    return (tables.symTbl != nullptr && tables.strTbl != nullptr);
}

/**
 * @brief Extract function symbols from ELF symbol table
 */
static void ExtractFunctionSymbols(const ElfSymbolTables& tables, SkFuncSymbolTable& funcSymTable) 
{
    size_t symCount = tables.symSize / sizeof(Elf64_Sym);
    for (size_t i = 0; i < symCount; ++i) {
        if ((tables.symTbl[i].st_info & 0xf) != STT_FUNC || tables.symTbl[i].st_size == 0) {
            continue;
        }
        const char* name = tables.strTbl + tables.symTbl[i].st_name;
        if (name == nullptr || name[0] == '\0') {
            continue;
        }
        SymBindType bindType = (tables.symTbl[i].st_info >> 4) == STB_WEAK ? SymBindType::WEAK
                              : (tables.symTbl[i].st_info >> 4) == STB_GLOBAL ? SymBindType::GLOBAL
                              : SymBindType::LOCAL;
        funcSymTable[tables.symTbl[i].st_value] = {name, tables.symTbl[i].st_size, bindType};
    }
}

SkFuncSymbolTable BuildFuncSymbolTable(const char* binAddr, size_t binSize) 
{
    SkFuncSymbolTable funcSymTable;
    if (binAddr == nullptr || binSize == 0) {
        SK_LOGE("Invalid bin parameters: binAddr=%p, binSize=%zu", binAddr, binSize);
        return funcSymTable;
    }
    ElfSymbolTables tables;
    if (!FindElfSymbolTables(binAddr, tables)) {
        SK_LOGE("Failed to find symtab or strtab sections");
        return funcSymTable;
    }
    ExtractFunctionSymbols(tables, funcSymTable);
    SK_LOGI("total %zu function symbols found", funcSymTable.size());
    return funcSymTable;
}

} // namespace

bool GetFuncSymbolInfo(const char* binAddr, size_t binSize, uint64_t funcAddr,
                       std::string& symbolName, uint64_t& funcSize, std::string& symbolBind)

{
    if (binAddr == nullptr || binSize == 0) {
        SK_LOGE("Invalid bin parameters: binAddr=%p, binSize=%zu", binAddr, binSize);
        return false;
    }
    static SkBinSymbolTable symbolTable;
    auto cacheIt = symbolTable.find(binAddr);
    if (cacheIt == symbolTable.end()) {
        SK_LOGI("Building symbol table for binAddr=%p", binAddr);
        symbolTable[binAddr] = BuildFuncSymbolTable(binAddr, binSize);
        cacheIt = symbolTable.find(binAddr);
    }

    const auto& funcSymTable = cacheIt->second;
    auto it = funcSymTable.find(funcAddr);
    if (it != funcSymTable.end()) {
        symbolName = it->second.name;
        funcSize = it->second.size;
        symbolBind = SymBindTypeToStr(it->second.bindType);
        SK_LOGI("Found symbol: name=%s, addr=0x%lx, size=0x%lx, bind=%s",
                symbolName.c_str(), funcAddr, funcSize, symbolBind.c_str());
        return true;
    }
    SK_LOGW("Function symbol not found for addr=0x%lx", funcAddr);
    return false;
}
