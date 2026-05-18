/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PGO_CONFIG_SETTERS_MIXIN_H_
#define ATT_PGO_CONFIG_SETTERS_MIXIN_H_

namespace att {

template <typename Derived>
class PgoConfigSettersMixin {
 public:
  void SetEnableMulticoreUBTradeoff(bool enable_multicore_ub_tradeoff) {
    enable_multicore_ub_tradeoff_ = enable_multicore_ub_tradeoff;
  }
  void SetEnableAutofusePGO(bool enable_autofuse_pgo) {
    enable_autofuse_pgo_ = enable_autofuse_pgo;
  }
  void SetIsInductorScene(bool is_inductor_scene) {
    is_inductor_scene_ = is_inductor_scene;
  }
  void SetAutofusePGOStepMax(int64_t pgo_step_max) {
    pgo_step_max_ = pgo_step_max;
  }

 protected:
  bool enable_multicore_ub_tradeoff_{false};
  bool enable_autofuse_pgo_{false};
  bool is_inductor_scene_{false};
  int64_t pgo_step_max_{16};
};

}  // namespace att

#endif  // ATT_PGO_CONFIG_SETTERS_MIXIN_H_