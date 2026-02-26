/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <elf.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "securec.h"
#include "rt_intf.h"

void get_meta_info(const char *elf, const char *fun, uint32_t *kernel_type, uint32_t ratio[2])
{
  char meta_name[256];
  const Elf64_Ehdr *ehdr = (const Elf64_Ehdr *)elf;
  uint64_t sh_cnt = ehdr->e_shnum;
  const Elf64_Shdr *sh_hdr = (const Elf64_Shdr *)(elf + ehdr->e_shoff);
  const char *sh_str_tbl = elf + sh_hdr[ehdr->e_shstrndx].sh_offset;
  printf("elf %p sh cnt %ld fun name %s\n", elf, sh_cnt, fun);
  int n_len = snprintf_s(meta_name, sizeof(meta_name), "%s.%s", ".ascend.meta", fun);
  if (n_len < 0 || (size_t)n_len >= sizeof(meta_name)) {
    printf("[error] snprintf_s failed or meta_name truncated\n");
    return;
  }
  for (int i = 0; i < sh_cnt; i++) {
    if (sh_hdr[i].sh_type == SHT_NULL ||
        sh_hdr[i].sh_type == SHT_NOBITS) {
      continue;
    }
    const char *sec_name = sh_str_tbl + sh_hdr[i].sh_name;
    printf("%s offset 0x%lx size 0x%lx\n", sec_name,
           sh_hdr[i].sh_offset, sh_hdr[i].sh_size);
    if (!strncmp(meta_name, sec_name, n_len) &&
        (sec_name[n_len] == 0 || !strcmp("_mix_aic", &sec_name[n_len]) ||
        !strcmp("_mix_aiv", &sec_name[n_len]))) {
      const char *meta_data = elf + sh_hdr[i].sh_offset;
      const uint16_t *tlv = (const uint16_t *)meta_data;
      while ((const char *)tlv - meta_data < sh_hdr[i].sh_size) {
        if (tlv[0] == 1) {
          *kernel_type = *(uint32_t *)(tlv + 2);
        } else if (tlv[0] == 3) {
          ratio[0] = tlv[2];
          ratio[1] = tlv[3];
        }
        tlv = (const uint16_t *)((const char *)tlv + 4 + tlv[1]);
      }
      break;
    }
  }
  printf("kernel %s type %u [%u:%u]\n", fun, *kernel_type, ratio[0], ratio[1]);
}

void print_kernel_symbols(const char *elf)
{
  char meta_name[256];
  const Elf64_Ehdr *ehdr = (const Elf64_Ehdr *)elf;
  uint64_t sh_cnt = ehdr->e_shnum;
  const Elf64_Shdr *sh_hdr = (const Elf64_Shdr *)(elf + ehdr->e_shoff);
  const char *sh_str_tbl = elf + sh_hdr[ehdr->e_shstrndx].sh_offset;
  const char *str_tbl = NULL;
  const Elf64_Sym *sym_tbl = NULL;
  size_t sym_size = 0;
  for (int i = 0; i < sh_cnt; i++) {
    if (sh_hdr[i].sh_type == SHT_NULL ||
        sh_hdr[i].sh_type == SHT_NOBITS) {
      continue;
    }
    const char *sec_name = sh_str_tbl + sh_hdr[i].sh_name;
    if (!strcmp(".symtab", sec_name)) {
      sym_tbl = (const Elf64_Sym *)(elf + sh_hdr[i].sh_offset);
      sym_size = sh_hdr[i].sh_size;
    } else if (!strcmp(".strtab", sec_name)) {
      str_tbl = elf + sh_hdr[i].sh_offset;
    }
  }
  if (!sym_tbl || !str_tbl) {
    printf("readelf error symtab %p, strtab %p\n", sym_tbl, str_tbl);
    return;
  }
  printf("offset   size     name\n");
  for (int i = 0; i < sym_size / sizeof(Elf64_Sym); i++) {
    if ((sym_tbl[i].st_info & 0xf) != STT_FUNC) {
      continue;
    }
    const char *name = str_tbl + sym_tbl[i].st_name;
    uint64_t value = sym_tbl[i].st_value;
    uint64_t size = sym_tbl[i].st_size;
    printf("%08lx %08lx %s\n", value, size, name);
  }
}

