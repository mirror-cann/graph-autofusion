/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <elf.h>

#define private public
#define protected public
#include "sk_common.h"

namespace {

class SkCommonTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::vector<uint8_t> BuildMinimalValidElf64()
{
    std::vector<uint8_t> buffer(sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3 + 128, 0);
    
    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(buffer.data());
    ehdr->e_ident[EI_MAG0] = ELFMAG0;
    ehdr->e_ident[EI_MAG1] = ELFMAG1;
    ehdr->e_ident[EI_MAG2] = ELFMAG2;
    ehdr->e_ident[EI_MAG3] = ELFMAG3;
    ehdr->e_ident[EI_CLASS] = ELFCLASS64;
    ehdr->e_shoff = sizeof(Elf64_Ehdr);
    ehdr->e_shnum = 3;
    ehdr->e_shstrndx = 2;
    
    Elf64_Shdr* shdr = reinterpret_cast<Elf64_Shdr*>(buffer.data() + ehdr->e_shoff);
    
    shdr[0].sh_name = 0;
    shdr[0].sh_type = SHT_NULL;
    
    shdr[1].sh_name = 1;
    shdr[1].sh_type = SHT_SYMTAB;
    shdr[1].sh_offset = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3;
    shdr[1].sh_size = sizeof(Elf64_Sym);
    
    shdr[2].sh_name = 10;
    shdr[2].sh_type = SHT_STRTAB;
    shdr[2].sh_offset = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3 + sizeof(Elf64_Sym);
    shdr[2].sh_size = 20;
    
    char* shstrtab = reinterpret_cast<char*>(buffer.data() + shdr[2].sh_offset);
    shstrtab[1] = '.';
    shstrtab[2] = 's';
    shstrtab[3] = 'y';
    shstrtab[4] = 'm';
    shstrtab[5] = 't';
    shstrtab[6] = 'a';
    shstrtab[7] = 'b';
    shstrtab[10] = '.';
    shstrtab[11] = 's';
    shstrtab[12] = 't';
    shstrtab[13] = 'r';
    shstrtab[14] = 't';
    shstrtab[15] = 'a';
    shstrtab[16] = 'b';
    
    Elf64_Sym* symtab = reinterpret_cast<Elf64_Sym*>(buffer.data() + shdr[1].sh_offset);
    symtab[0].st_name = 0;
    symtab[0].st_value = 0;
    symtab[0].st_size = 0;
    symtab[0].st_info = 0;
    
    char* strtab = reinterpret_cast<char*>(buffer.data() + shdr[2].sh_offset + shdr[2].sh_size);
    
    return buffer;
}

} // namespace

TEST_F(SkCommonTest, SkKernelArchToString_ReturnsExpectedName)
{
    EXPECT_STREQ(to_string(SkKernelArch::DAV_2201), "DAV_2201");
    EXPECT_STREQ(to_string(SkKernelArch::DAV_3510), "DAV_3510");
    EXPECT_STREQ(to_string(SkKernelArch::UNKNOWN), "UNKNOWN");
}

TEST_F(SkCommonTest, GetSkKernelArchSymbolSuffix_ReturnsExpectedSuffix)
{
    EXPECT_STREQ(GetSkKernelArchSymbolSuffix(SkKernelArch::DAV_2201), "dav_2201");
    EXPECT_STREQ(GetSkKernelArchSymbolSuffix(SkKernelArch::DAV_3510), "dav_3510");
    EXPECT_STREQ(GetSkKernelArchSymbolSuffix(SkKernelArch::UNKNOWN), "unknown");
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_NullBinAddr_ReturnsFalse)
{
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), nullptr, 100, 0x10,
                                  symbolName, funcSize, symbolBind);
    EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_ZeroBinSize_ReturnsFalse)
{
    char buffer[100] = {0};
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), buffer, 0, 0x10,
                                  symbolName, funcSize, symbolBind);
    EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_BinSizeSmallerThanElfHeader_ReturnsFalse)
{
    char buffer[10] = {0};
    buffer[0] = ELFMAG0;
    buffer[1] = ELFMAG1;
    buffer[2] = ELFMAG2;
    buffer[3] = ELFMAG3;
    
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000), buffer, 10, 0x10,
                                  symbolName, funcSize, symbolBind);
    EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_SectionHeaderExceedsBinSize_ReturnsFalse)
{
    size_t bufferSize = sizeof(Elf64_Ehdr);
    std::vector<uint8_t> buffer(bufferSize, 0);
    
    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(buffer.data());
    ehdr->e_ident[EI_MAG0] = ELFMAG0;
    ehdr->e_ident[EI_MAG1] = ELFMAG1;
    ehdr->e_ident[EI_MAG2] = ELFMAG2;
    ehdr->e_ident[EI_MAG3] = ELFMAG3;
    ehdr->e_ident[EI_CLASS] = ELFCLASS64;
    ehdr->e_shoff = 1000;
    ehdr->e_shnum = 10;
    ehdr->e_shstrndx = 0;
    
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000),
                                  reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                  symbolName, funcSize, symbolBind);
    EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_SectionHeaderTableSizeExceedsBinSize_ReturnsFalse)
{
    size_t bufferSize = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr);
    std::vector<uint8_t> buffer(bufferSize, 0);
    
    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(buffer.data());
    ehdr->e_ident[EI_MAG0] = ELFMAG0;
    ehdr->e_ident[EI_MAG1] = ELFMAG1;
    ehdr->e_ident[EI_MAG2] = ELFMAG2;
    ehdr->e_ident[EI_MAG3] = ELFMAG3;
    ehdr->e_ident[EI_CLASS] = ELFCLASS64;
    ehdr->e_shoff = sizeof(Elf64_Ehdr);
    ehdr->e_shnum = 100;
    ehdr->e_shstrndx = 0;
    
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000),
                                  reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                  symbolName, funcSize, symbolBind);
    EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_InvalidShstrndx_ReturnsFalse)
{
    size_t bufferSize = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 2;
    std::vector<uint8_t> buffer(bufferSize, 0);
    
    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(buffer.data());
    ehdr->e_ident[EI_MAG0] = ELFMAG0;
    ehdr->e_ident[EI_MAG1] = ELFMAG1;
    ehdr->e_ident[EI_MAG2] = ELFMAG2;
    ehdr->e_ident[EI_MAG3] = ELFMAG3;
    ehdr->e_ident[EI_CLASS] = ELFCLASS64;
    ehdr->e_shoff = sizeof(Elf64_Ehdr);
    ehdr->e_shnum = 2;
    ehdr->e_shstrndx = 10;
    
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000),
                                  reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                  symbolName, funcSize, symbolBind);
    EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_ShstrtabExceedsBinSize_ReturnsFalse)
{
    size_t bufferSize = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3;
    std::vector<uint8_t> buffer(bufferSize, 0);
    
    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(buffer.data());
    ehdr->e_ident[EI_MAG0] = ELFMAG0;
    ehdr->e_ident[EI_MAG1] = ELFMAG1;
    ehdr->e_ident[EI_MAG2] = ELFMAG2;
    ehdr->e_ident[EI_MAG3] = ELFMAG3;
    ehdr->e_ident[EI_CLASS] = ELFCLASS64;
    ehdr->e_shoff = sizeof(Elf64_Ehdr);
    ehdr->e_shnum = 3;
    ehdr->e_shstrndx = 2;
    
    Elf64_Shdr* shdr = reinterpret_cast<Elf64_Shdr*>(buffer.data() + ehdr->e_shoff);
    shdr[2].sh_offset = 10000;
    shdr[2].sh_size = 100;
    
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000),
                                  reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                  symbolName, funcSize, symbolBind);
    EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_MissingSymtabAndStrtab_ReturnsFalse)
{
    size_t bufferSize = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3 + 50;
    std::vector<uint8_t> buffer(bufferSize, 0);
    
    Elf64_Ehdr* ehdr = reinterpret_cast<Elf64_Ehdr*>(buffer.data());
    ehdr->e_ident[EI_MAG0] = ELFMAG0;
    ehdr->e_ident[EI_MAG1] = ELFMAG1;
    ehdr->e_ident[EI_MAG2] = ELFMAG2;
    ehdr->e_ident[EI_MAG3] = ELFMAG3;
    ehdr->e_ident[EI_CLASS] = ELFCLASS64;
    ehdr->e_shoff = sizeof(Elf64_Ehdr);
    ehdr->e_shnum = 3;
    ehdr->e_shstrndx = 2;
    
    Elf64_Shdr* shdr = reinterpret_cast<Elf64_Shdr*>(buffer.data() + ehdr->e_shoff);
    shdr[0].sh_type = SHT_NULL;
    shdr[1].sh_type = SHT_PROGBITS;
    shdr[2].sh_type = SHT_STRTAB;
    shdr[2].sh_offset = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * 3;
    shdr[2].sh_size = 50;
    
    char* shstrtab = reinterpret_cast<char*>(buffer.data() + shdr[2].sh_offset);
    shstrtab[1] = '.';
    shstrtab[2] = 't';
    shstrtab[3] = 'e';
    shstrtab[4] = 'x';
    shstrtab[5] = 't';
    
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1000),
                                  reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                  symbolName, funcSize, symbolBind);
    EXPECT_FALSE(ret);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_DifferentBinHandleBuildsSeparateCache)
{
    auto buffer = BuildMinimalValidElf64();
    
    std::string symbolName;
    uint64_t funcSize = 0;
    std::string symbolBind;
    
    bool ret1 = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1001),
                                   reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                   symbolName, funcSize, symbolBind);
    
    bool ret2 = GetFuncSymbolInfo(reinterpret_cast<aclrtBinHandle>(0x1002),
                                   reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                   symbolName, funcSize, symbolBind);
    
    EXPECT_FALSE(ret1);
    EXPECT_FALSE(ret2);
}

TEST_F(SkCommonTest, GetFuncSymbolInfo_SameBinHandleUsesCache)
{
    auto buffer = BuildMinimalValidElf64();
    
    std::string symbolName1, symbolName2;
    uint64_t funcSize1 = 0, funcSize2 = 0;
    std::string symbolBind1, symbolBind2;
    
    aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(0x2001);
    
    bool ret1 = GetFuncSymbolInfo(binHdl,
                                  reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                  symbolName1, funcSize1, symbolBind1);
    
    bool ret2 = GetFuncSymbolInfo(binHdl,
                                  reinterpret_cast<char*>(buffer.data()), buffer.size(), 0x10,
                                  symbolName2, funcSize2, symbolBind2);
    
    EXPECT_EQ(ret1, ret2);
    EXPECT_EQ(symbolName1, symbolName2);
    EXPECT_EQ(funcSize1, funcSize2);
    EXPECT_EQ(symbolBind1, symbolBind2);
}
