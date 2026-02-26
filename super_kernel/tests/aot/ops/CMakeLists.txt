# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_subdirectory(interface)
add_subdirectory(rms_norm)
add_subdirectory(grouped_matmul)
add_subdirectory(weight_quant_batch_matmul_v2)
add_subdirectory(matmul_add)
add_subdirectory(dequant_swiglu_quant)
add_subdirectory(clear_ops)
add_subdirectory(grouped_matmul_v2)
add_subdirectory(grouped_matmul_v3)
add_subdirectory(dynamic_quant)