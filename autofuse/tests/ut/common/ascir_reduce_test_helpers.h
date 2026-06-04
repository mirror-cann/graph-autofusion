/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_COMMON_ASCIR_REDUCE_TEST_HELPERS_H_
#define TESTS_UT_COMMON_ASCIR_REDUCE_TEST_HELPERS_H_

#include <memory>
#include <string>
#include <vector>

#include "ascir_ops.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "gen_model_info/parser/reduce_specific_params_builder.h"
#include "gen_model_info/parser/tuning_space.h"

namespace ascir_reduce_test_helpers {

struct ReduceTestEnv {
  af::AscGraph graph;
  ge::Expression s0;
  ge::Expression s1;
  af::Axis z0;
  af::Axis z1;
  af::AscNodePtr node;
  att::SubAxis axis0;
  att::SubAxis axis1;

  explicit ReduceTestEnv(const char *reduce_name)
      : graph("reduce_test_graph")
  {
    s0 = graph.CreateSizeVar("s0");
    s1 = graph.CreateSizeVar("s1");
    z0 = graph.CreateAxis("z0", s0);
    z1 = graph.CreateAxis("z1", s1);
    af::ascir_op::Data x("x", graph);
    af::ascir_op::Load load("load");
    af::ascir_op::Max reduce_op(reduce_name);
    graph.AddNode(load);
    graph.AddNode(reduce_op);
    load.x = x.y;
    reduce_op.x = load.y;
    node = graph.FindNode(reduce_name);
    node->attr.sched.axis = {z0.id, z1.id};
    node->attr.sched.loop_axis = z1.id;
    node->inputs[0].attr.dtype = ge::DT_FLOAT16;
    node->inputs[0].attr.axis = {z0.id, z1.id};
    node->inputs[0].attr.repeats = {s0, s1};
    node->outputs[0].attr.dtype = ge::DT_FLOAT16;
    node->outputs[0].attr.axis = {z0.id, z1.id};
    axis0.id = z0.id;
    axis0.name = "z0";
    axis0.repeat = s0;
    axis1.id = z1.id;
    axis1.name = "z1";
    axis1.repeat = s1;
  }

  void SetIoAttrs(const std::vector<ge::Expression> &in_strides,
                  const std::vector<ge::Expression> &out_repeats,
                  const std::vector<ge::Expression> &out_strides)
  {
    node->inputs[0].attr.strides = in_strides;
    node->inputs[0].attr.vectorized_axis = {z0.id, z1.id};
    node->inputs[0].attr.vectorized_strides = in_strides;
    node->outputs[0].attr.repeats = out_repeats;
    node->outputs[0].attr.strides = out_strides;
    node->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
    node->outputs[0].attr.vectorized_strides = out_strides;
  }
};

inline att::TensorPtr BuildParserTensor(const std::string &name, const std::vector<att::SubAxis *> &axes,
                                        const std::vector<ge::Expression> &repeats,
                                        const std::vector<ge::Expression> &strides)
{
  auto tensor = std::make_shared<att::Tensor>();
  tensor->name = name;
  tensor->data_type_size = 2U;
  tensor->dim_info = axes;
  tensor->repeat = repeats;
  tensor->stride = strides;
  return tensor;
}

inline att::NodeInfo BuildReduceNodeInfo(ReduceTestEnv &env, const std::string &node_name)
{
  att::NodeInfo node_info;
  node_info.name = node_name;
  node_info.node_type = "Max";
  node_info.node_ptr = env.node;
  node_info.inputs = {BuildParserTensor("x", {&env.axis0, &env.axis1}, {ge::Symbol(8), ge::Symbol(16)},
                                        {ge::Symbol(16), ge::Symbol(1)})};
  node_info.outputs = {BuildParserTensor("y", {&env.axis0, &env.axis1}, {ge::Symbol(8), ge::Symbol(1)},
                                         {ge::Symbol(1), ge::Symbol(0)})};
  node_info.loop_axes = {&env.axis1};
  (void)ascir_param::EnrichAscirGraphNodeParams(env.graph);
  (void)att::FillReduceSpecificParams(env.node, node_info);
  return node_info;
}

}  // namespace ascir_reduce_test_helpers

#endif  // TESTS_UT_COMMON_ASCIR_REDUCE_TEST_HELPERS_H_
