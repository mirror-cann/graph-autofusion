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
using SkBinSymbolTable = std::unordered_map<aclrtBinHandle, SkFuncSymbolTable>;

struct ElfSymbolTables {
    const Elf64_Sym* symTbl;
    size_t symSize;
    const char* strTbl;
    size_t strTblSize;
};

static bool ValidateElfSectionRange(size_t binSize, uint64_t offset, uint64_t size)
{
    return offset <= binSize && offset + size <= binSize;
}

static bool FindElfSymbolTables(const char* binAddr, size_t binSize, ElfSymbolTables& tables) {
    constexpr size_t ELF64_EHDR_SIZE = sizeof(Elf64_Ehdr);
    constexpr size_t ELF64_SHDR_SIZE = sizeof(Elf64_Shdr);
    
    if (binSize < ELF64_EHDR_SIZE) {
        SK_LOGE("Invalid ELF: binSize=%zu < minimum header size %zu", binSize, ELF64_EHDR_SIZE);
        return false;
    }
    
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(binAddr);
    
    if (ehdr->e_shoff > binSize ||
        ehdr->e_shoff + static_cast<uint64_t>(ehdr->e_shnum) * ELF64_SHDR_SIZE > binSize) {
        SK_LOGE("Invalid ELF: shoff=0x%lx, shnum=%u exceed binSize=%zu", 
                static_cast<uint64_t>(ehdr->e_shoff), ehdr->e_shnum, binSize);
        return false;
    }
    
    if (ehdr->e_shstrndx >= ehdr->e_shnum) {
        SK_LOGE("Invalid ELF: e_shstrndx=%u >= e_shnum=%u", ehdr->e_shstrndx, ehdr->e_shnum);
        return false;
    }
    
    const Elf64_Shdr* shHdr = reinterpret_cast<const Elf64_Shdr*>(binAddr + ehdr->e_shoff);
    
    const Elf64_Shdr& shstrtabHdr = shHdr[ehdr->e_shstrndx];
    if (!ValidateElfSectionRange(binSize, shstrtabHdr.sh_offset, shstrtabHdr.sh_size)) {
        SK_LOGE("Invalid ELF: shstrtab offset=0x%lx size=0x%lx exceed binSize", 
                static_cast<uint64_t>(shstrtabHdr.sh_offset), static_cast<uint64_t>(shstrtabHdr.sh_size));
        return false;
    }
    
    const char* shStrTbl = binAddr + shstrtabHdr.sh_offset;
    size_t shStrTblSize = shstrtabHdr.sh_size;
    
    tables.symTbl = nullptr;
    tables.symSize = 0;
    tables.strTbl = nullptr;
    tables.strTblSize = 0;
    
    for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
        if (shHdr[i].sh_type == SHT_NULL || shHdr[i].sh_type == SHT_NOBITS) {
            continue;
        }
        
        if (shHdr[i].sh_name >= shStrTblSize) {
            continue;
        }
        
        const char* secName = shStrTbl + shHdr[i].sh_name;
        size_t secNameMaxLen = shStrTblSize - shHdr[i].sh_name;
        
        if (!ValidateElfSectionRange(binSize, shHdr[i].sh_offset, shHdr[i].sh_size)) {
            continue;
        }
        
        if (strncmp(".symtab", secName, secNameMaxLen) == 0) {
            tables.symTbl = reinterpret_cast<const Elf64_Sym*>(binAddr + shHdr[i].sh_offset);
            tables.symSize = shHdr[i].sh_size;
        } else if (strncmp(".strtab", secName, secNameMaxLen) == 0) {
            tables.strTbl = binAddr + shHdr[i].sh_offset;
            tables.strTblSize = shHdr[i].sh_size;
        }
    }
    return (tables.symTbl != nullptr && tables.strTbl != nullptr);
}

static void ExtractFunctionSymbols(const ElfSymbolTables& tables, SkFuncSymbolTable& funcSymTable) 
{
    constexpr size_t ELF64_SYM_SIZE = sizeof(Elf64_Sym);
    size_t symCount = tables.symSize / ELF64_SYM_SIZE;
    
    for (size_t i = 0; i < symCount; ++i) {
        const Elf64_Sym& sym = tables.symTbl[i];
        
        if ((sym.st_info & 0xf) != STT_FUNC || sym.st_size == 0) {
            continue;
        }
        
        if (sym.st_name >= tables.strTblSize) {
            continue;
        }
        
        const char* name = tables.strTbl + sym.st_name;
        if (name == nullptr || name[0] == '\0') {
            continue;
        }
        
        SymBindType bindType = (sym.st_info >> 4) == STB_WEAK ? SymBindType::WEAK
                              : (sym.st_info >> 4) == STB_GLOBAL ? SymBindType::GLOBAL
                              : SymBindType::LOCAL;
        funcSymTable[sym.st_value] = {name, sym.st_size, bindType};
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
    if (!FindElfSymbolTables(binAddr, binSize, tables)) {
        SK_LOGE("Failed to find valid symtab or strtab sections");
        return funcSymTable;
    }
    ExtractFunctionSymbols(tables, funcSymTable);
    SK_LOGI("total %zu function symbols found", funcSymTable.size());
    return funcSymTable;
}

} // namespace

bool GetFuncSymbolInfo(aclrtBinHandle binHdl, const char* binAddr, size_t binSize, uint64_t funcAddr,
                       std::string& symbolName, uint64_t& funcSize, std::string& symbolBind)

{
    if (binAddr == nullptr || binSize == 0) {
        SK_LOGE("Invalid bin parameters: binAddr=%p, binSize=%zu", binAddr, binSize);
        return false;
    }
    static SkBinSymbolTable symbolTable;
    auto cacheIt = symbolTable.find(binHdl);
    if (cacheIt == symbolTable.end()) {
        SK_LOGI("Building symbol table for binHdl=%p", binHdl);
        symbolTable[binHdl] = BuildFuncSymbolTable(binAddr, binSize);
        cacheIt = symbolTable.find(binHdl);
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
