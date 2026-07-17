#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

import npu_bridge

_ = npu_bridge  # 通过import副作用注册NPU算子

if __name__ == "__main__":
    data1 = tf.placeholder(tf.float16, shape=[128, 192])
    input_data = np.random.rand(128, 192).astype(np.float16)
    ## 构造模型结构
    abs_0 = tf.abs(data1)
    relu_0 = tf.nn.relu(abs_0)
    exp_0 = tf.exp(relu_0)

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    # 设置为推理模式
    custom_op.parameter_map["graph_run_mode"].i = 0
    # 开启 profiling（等价于 inductor 场景的 torch_npu.profiler）
    profiling_options = (
        '{"output":"./profiling","training_trace":"on","task_time":"on",'
        '"hccl":"on","aicpu":"on","aic_metrics":"PipeUtilization","msproftx":"off"}'
    )
    custom_op.parameter_map["profiling_mode"].b = True
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        profiling_options
    )
    # 准备输入数据
    feed_dict = {data1: input_data}
    step = 100
    ## 执行模型
    with tf.compat.v1.Session(config=sess_config) as sess:
        for _ in range(step):
            result = sess.run(exp_0, feed_dict=feed_dict)
