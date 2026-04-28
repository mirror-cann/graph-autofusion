/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_PRE_PROCESS_PRE_PROCESS_CONFIG_H
#define AUTOFUSE_OPTIMIZE_PRE_PROCESS_PRE_PROCESS_CONFIG_H

#include <cctype>
#include <cstdlib>
#include <string>
#include <unordered_set>

namespace af {
namespace pre_process {

class PreProcessConfig {
 public:
  static PreProcessConfig &Instance() {
    static PreProcessConfig config;
    return config;
  }

  const std::unordered_set<std::string> &GetImprovePrecisionBlacklist() const { return blacklist_; }

  void Reset() {
    blacklist_.clear();
    ParseBlacklist();
  }

  PreProcessConfig(const PreProcessConfig &) = delete;
  PreProcessConfig &operator=(const PreProcessConfig &) = delete;

 private:
  PreProcessConfig() { ParseBlacklist(); }

  // 与前端 ReadImprovePrecisionBlacklist 保持一致的解析逻辑
  static std::unordered_set<std::string> ReadImprovePrecisionBlacklist(std::string &input) {
    std::unordered_set<std::string> tokens;
    if (!input.empty() && std::ispunct(static_cast<unsigned char>(input.back())) != 0) {
      input.pop_back();
    }
    size_t start = 0U;
    size_t end = input.find(',');
    while (end != std::string::npos) {
      tokens.insert(input.substr(start, end - start));
      start = end + 1U;
      end = input.find(',', start);
    }
    tokens.insert(input.substr(start));
    return tokens;
  }

  static std::string ParseEnvFlag(const std::string &env_str, const std::string &key) {
    auto pos = env_str.find(key);
    if (pos == std::string::npos) {
      return "";
    }
    std::string value = env_str.substr(pos + key.size());
    auto end_pos = value.find(';');
    if (end_pos != std::string::npos) {
      value = value.substr(0, end_pos);
    }
    return value;
  }

  void ParseBlacklist() {
    const char *autofuse_env = std::getenv("AUTOFUSE_FLAGS");
    if ((autofuse_env == nullptr) || (autofuse_env[0] == '\0')) {
      return;
    }
    std::string env_str(autofuse_env);
    const std::string key = "--autofuse_enhance_precision_blacklist=";
    std::string value = ParseEnvFlag(env_str, key);
    if (value.empty()) {
      return;
    }
    auto parsed = ReadImprovePrecisionBlacklist(value);
    blacklist_.insert(parsed.begin(), parsed.end());
  }

  std::unordered_set<std::string> blacklist_;
};

}  // namespace pre_process
}  // namespace af

#endif  // AUTOFUSE_OPTIMIZE_PRE_PROCESS_PRE_PROCESS_CONFIG_H
