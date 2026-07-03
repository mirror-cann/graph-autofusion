/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_TESTS_V35_UT_AUTOFUSE_LOOP_TEST_UTILS_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_TESTS_V35_UT_AUTOFUSE_LOOP_TEST_UTILS_H_

#include <string>
#include <type_traits>
#include <vector>

namespace af {
namespace loop {

template <typename T, typename F>
std::string StrJoin(const std::vector<T> &vec, F f, const std::string &sep = ", ") {
  if (vec.empty()) {
    return "[]";
  }
  std::string res = "[" + f(vec[0]);
  for (size_t i = 1U; i < vec.size(); ++i) {
    res += sep + f(vec[i]);
  }
  return res + "]";
}

inline std::string StrJoin(const std::vector<std::string> &vec, const std::string &sep = ", ") {
  return StrJoin(vec, [](const std::string &s) { return s; }, sep);
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, std::string>::type StrJoin(
    const std::vector<T> &vec, const std::string &sep = ", ") {
  return StrJoin(vec, [](const T &s) { return std::to_string(s); }, sep);
}

}  // namespace loop
}  // namespace af

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_TESTS_V35_UT_AUTOFUSE_LOOP_TEST_UTILS_H_
